from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from models.generator import CrossAttenGenerator
from image_transformer import rotation
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
Image.MAX_IMAGE_PIXELS = None
import torch.optim as optim
import numpy as np

parser = argparse.ArgumentParser(description='Clip-based Generative Networks')
parser.add_argument('--train_dir', default='./dataset/ImageNet/train', help='imagenet')
parser.add_argument('--batch_size', type=int, default=20, help='Number of training samples/batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
parser.add_argument('--eps', type=int, default=16, help='Perturbation budget')
parser.add_argument('--model_type', type=str, default='res152', help='Source model')
parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
parser.add_argument('--label_flag', type=str, default='N8', help='Label nums: N8, C20,...,C200')
parser.add_argument('--nz', type=int, default=16, help='nz')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='Dictionary to save the model')
parser.add_argument('--load_path', type=str, help='Path to checkpoint')
parser.add_argument('--finetune', action='store_true', help='Finetune for single class attack')
parser.add_argument('--finetune_class', type=int, help='Class id to be finetuned')
parser.add_argument('--mask_ratio', type=float, default='2e-1', help='Mask ratio in finetune stage')
args = parser.parse_args()
print(args)

# set class
n_class = 1000

# save loss record
all_losses = []

# Normalize (0-1)
eps = args.eps / 255.
use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(42)

# GPU
device_ids = [i for i in range(0, torch.cuda.device_count())]
print(device_ids)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Input dimension and generator
if args.model_type == 'incv3':
    scale_size, img_size = 300, 299
    netG = CrossAttenGenerator(inception=True, nz=args.nz, device=device)
else:
    scale_size, img_size = 256, 224
    netG = CrossAttenGenerator(nz=args.nz, device=device)

if args.load_path is not None:
    print(f"Loading checkpoint from {args.load_path}")
    # 这里直接用 netG 加载（无 DataParallel 的情况下），不分单卡多卡
    checkpoint = torch.load(args.load_path, map_location=device)
    #print("Checkpoint keys:", checkpoint.keys())
    netG.load_state_dict(checkpoint['model_state_dict'])
    # 优化器也类似
    # optimG.load_state_dict(checkpoint['optimizer_state_dict'])  # 优化器在下面初始化后再加载
    start_epoch = checkpoint.get('epoch', args.start_epoch)
    start_batch = checkpoint.get('batch', 0)
    print(f"Resumed from epoch {start_epoch}, batch {start_batch}")
else:
    print("Training from scratch!")
    start_epoch = args.start_epoch    # ★★★ 这里补上这一句 ★★★
    start_batch = 0 

# 2. DataParallel 包裹
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    netG = nn.DataParallel(netG, device_ids=device_ids)
netG = netG.to(device)

# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
if args.load_path is not None:
    optimG.load_state_dict(checkpoint['optimizer_state_dict'])

if torch.cuda.device_count() > 1:
    optimG = nn.DataParallel(optimG, device_ids=device_ids)
    optimG = optimG.module

# Data
train_set = get_data(args.train_dir, scale_size, img_size)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True)

# Surrogate model
if args.model_type == 'incv3':
    model = torchvision.models.inception_v3(pretrained=True).to(device)
elif args.model_type == 'res152':
    model = torchvision.models.resnet152(pretrained=True).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.module
model.eval()

# class
label_set = get_classes(args.label_flag)

# Loss
criterion = nn.CrossEntropyLoss()

# text condition
text_cond_dict = torch.load('text_feature.pth')

# save dir
save_dir = os.path.join(args.save_dir, args.model_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# loss save dir
loss_save_dir = os.path.join(os.getcwd(), "loss")
if not os.path.exists(loss_save_dir):
    os.makedirs(loss_save_dir)

# Training
for epoch in range(start_epoch, args.epochs):
    running_loss = 0
    for i, (imgs, _) in enumerate(tqdm(train_loader)):
        if epoch == start_epoch and i < start_batch:
            continue  # 跳过已训练的batch
        img = imgs[0].to(device)
        img_rot = rotation(img)[0]
        img_aug = imgs[1].to(device)
        if args.finetune:
            label = np.array([args.finetune_class] * img.size(0))
        else:
            np.random.shuffle(label_set)
            label = np.random.choice(label_set, img.size(0))
        cond = torch.stack([text_cond_dict[j] for j in label], dim=0)
        label = torch.from_numpy(label).long().to(device)
        netG.train()
        optimG.zero_grad()

        # generate img
        noise = netG(input=img, cond=cond, eps=eps)
        noise_rot = netG(input=img_rot, cond=cond, eps=eps)
        noise_aug = netG(input=img_aug, cond=cond, eps=eps)

        if args.finetune:
            noise = get_mask(noise, args.mask_ratio, device)
            noise_rot = get_mask(noise_rot, args.mask_ratio, device)
            noise_aug = get_mask(noise_aug, args.mask_ratio, device)

        adv = noise + img
        adv = torch.clamp(adv, 0.0, 1.0)

        adv_rot = noise_rot + img_rot
        adv_rot = torch.clamp(adv_rot, 0.0, 1.0)

        adv_aug = noise_aug + img_aug
        adv_aug = torch.clamp(adv_aug, 0.0, 1.0)

        adv_out = model(normalize(adv))
        adv_rot_out = model(normalize(adv_rot))
        adv_aug_out = model(normalize(adv_aug))

        loss = criterion(adv_out, label) + criterion(adv_rot_out, label) + criterion(adv_aug_out, label)
        loss.backward()
        optimG.step()

        # 新增：保存每个batch的loss
        all_losses.append(loss.item())
        if i % 5000 == 0 and i > 0:
            np.save(os.path.join(loss_save_dir, "all_losses.npy"), np.array(all_losses))

        if i % 10 == 9:
            print('Epoch: {} \t Batch: {}/{} \t loss: {:.5f}'.format(epoch, i, len(train_loader), running_loss / 100))
            running_loss = 0
        running_loss += abs(loss.item())
        # 每5000个batch保存一次
        if i % 5000 == 0 and i > 0:
            ckpt_name = '{}/model-epoch{}_batch{}.pth'.format(save_dir, epoch, i)
            # 保存完整状态，包括epoch和batch号、模型和优化器
            state = {
                'epoch': epoch,
                'batch': i,
                'model_state_dict': netG.module.state_dict() if torch.cuda.device_count() > 1 else netG.state_dict(),
                'optimizer_state_dict': optimG.state_dict(),
            }
            torch.save(state, ckpt_name)
            print(f"[Checkpoint] Saved at {ckpt_name}")
    if epoch >= start_epoch:
        ckpt_name = '{}/model-{}.pth'.format(save_dir, epoch)
        state = {
            'epoch': epoch,
            'batch': 0,
            'model_state_dict': netG.module.state_dict() if torch.cuda.device_count() > 1 else netG.state_dict(),
            'optimizer_state_dict': optimG.state_dict(),
        }
        torch.save(state, ckpt_name)

# 新增：训练全部结束后保存loss到文件
np.save(os.path.join(loss_save_dir, "all_losses.npy"), np.array(all_losses))
# 你也可以保存成txt或csv格式，比如：
# np.savetxt(os.path.join(loss_save_dir, "all_losses.txt"), np.array(all_losses))
print(f"All loss values saved to {os.path.join(loss_save_dir, 'all_losses.npy')}")