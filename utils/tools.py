import torchvision.transforms as transforms
from PIL import Image

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def reconstruct_image(input_tensor):
    img = input_tensor[0]
    unnormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(MEAN["imagenet"], STD["imagenet"])],
        std=[1 / s for s in STD["imagenet"]]
    )
    transformed_img = unnormalize(img)

    # 转换为 PIL 格式以保存
    # (C, H, W) -> (H, W, C) 并恢复到 0-255 范围
    transformed_img = transformed_img.permute(1, 2, 0).clamp(0, 1)  # 保证值在 [0, 1]
    transformed_img = (transformed_img * 255).byte().cpu().numpy()
    transformed_img = Image.fromarray(transformed_img)

    # 保存图像
    save_path = "reconstructed_image.jpg"
    transformed_img.save(save_path)