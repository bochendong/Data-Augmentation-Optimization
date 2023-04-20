
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


def extract_features(images, pre_trian_vgg):
    with torch.no_grad():
        features = pre_trian_vgg(images)
    return features.view(images.size(0), -1)

def find_nearest_neighbor(images, target_label, features_dict, pre_trian_vgg):
    target_label_int = target_label.item()
    target_images = features_dict[target_label_int]['images']
    target_features = features_dict[target_label_int]['features']

    # Convert the MNIST images to 3-channel format for VGG model
    images_3channel = images.repeat(1, 3, 1, 1)

    # Extract features for input images
    input_features = extract_features(images_3channel, pre_trian_vgg)

    # Compute distances between input_features and target_features
    input_features_expanded = input_features.unsqueeze(1)
    target_features_expanded = target_features.unsqueeze(0)

    # Compute distances between input images and target_images
    distances = (input_features_expanded - target_features_expanded).view(images.size(0), -1, target_features.size(1)).norm(dim=2)

    # Find the index of the input image with the smallest distance to the selected target_image
    min_distances, min_indices = distances.min(dim=1)
    closest_input_image_index = min_indices[min_distances.argmin()]

    return target_images[min_indices[closest_input_image_index]]

def generate_synthetic_digits(train_dataset, digit, count, flag = 1):
    if flag == 0:
        digit_indices = np.where(train_dataset.targets.cpu() == digit)[0]
    else:
        digit_indices = np.where(train_dataset.targets.cpu() == digit.cpu())[0]
    
    if len(digit_indices) == 0:
        raise ValueError(f"No samples found for label {digit.item()}")
        
    selected_indices = np.random.choice(digit_indices, count, replace=True)
    synthetic_digits = torch.stack([train_dataset[i][0] for i in selected_indices])
    return synthetic_digits

def build_features_dict(train_dataset, pre_trian_vgg, BATCH_SIZE, device):
    features_dict = {}
    for digit in range(10):
        synthetic_digits = generate_synthetic_digits(train_dataset, digit, BATCH_SIZE, 0)
        synthetic_digits_3channel = synthetic_digits.repeat(1, 3, 1, 1).to(device)
        synthetic_features = extract_features(synthetic_digits_3channel, pre_trian_vgg)
        features_dict[digit] = {'images': synthetic_digits, 'features': synthetic_features}
    return features_dict


# Erode the input images to remove the digit information
def erode_images(images):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_images = []
    for image in images:
        gray_image = image.squeeze(0).detach().cpu().numpy()
        eroded_image = cv2.erode(gray_image, kernel, iterations=1)
        eroded_images.append(eroded_image)
    
    eroded_images_np = np.array(eroded_images)
    return torch.tensor(eroded_images_np).unsqueeze(1).cuda()
def write_log(epoch, num_epochs, stats, log_dir="logs"):
    # Unpack the dictionary
    unet_loss = stats['unet_loss']
    fake_loss = stats['fake_loss']
    real_loss = stats['real_loss']
    unet_losses = stats['unet_losses']
    fake_losses = stats['fake_losses']
    real_losses = stats['real_losses']

    # Append the losses to the corresponding lists
    unet_losses.append(unet_loss)
    fake_losses.append(fake_loss)
    real_losses.append(real_loss)

    log_message = f"Epoch: {epoch+1}/{num_epochs}\n"
    log_message += f"Unet loss: {unet_loss:.4f}\n"
    log_message += f"Synthetic CNN loss: {fake_loss:.4f}, Real CNN loss: {real_loss:.4f}\n"
    print(log_message)
    
    with open(os.path.join(log_dir, "training_logs.txt"), "a") as log_file:
        log_file.write(log_message)

    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        fig, axes = plt.subplots(1, 3, figsize=(10, 10))

        axes[0, 0].plot(unet_losses)
        axes[0, 0].set_title("Unet Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")

        axes[0, 1].plot(fake_losses)
        axes[0, 1].set_title("Fake Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")

        axes[0, 1].plot(real_losses)
        axes[0, 2].set_title("Real Loss")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Loss")


        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"loss_curves_epoch_{epoch+1}.png"), dpi=300)
        plt.close(fig)

def loss_info(step, total_length, epoch, unet_loss, fake_loss, real_loss, fake_img, real_img, target_img):
    print(f'Step [{step + 1}/{total_length}]', end = ', ')
    print('unet_loss:' , unet_loss.data.detach().cpu().numpy(),  
                      'fake_loss:', fake_loss.data.detach().cpu().numpy(), 
                      'real_loss:', real_loss.detach().cpu().numpy())
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))

    # Display the real image
    ax1.imshow(real_img, cmap='gray')
    ax1.set_title("Real Image")
    ax1.axis("off")

    # Display the fake image
    ax2.imshow(fake_img, cmap='gray')
    ax2.set_title("Fake Image")
    ax2.axis("off")

    # Display the target image
    ax3.imshow(target_img, cmap='gray')
    ax3.set_title("Target Image")
    ax3.axis("off")

    plt.savefig('./output/%03d/%04d_recon.png' % ( epoch, step))
    plt.close()