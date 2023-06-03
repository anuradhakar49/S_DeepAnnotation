class CustomDataset2():
    """ CustomDataset : Class that loads data (images and masks) in efficent way"""
    def __init__(self, imgs_dirs, masks_dirs, ref_image_path, normalize=False,cached_data=True, n_channels=1,scale=1):
        self.imgs_dirs = imgs_dirs    # All paths to images 
        self.masks_dirs = masks_dirs  # All paths to masks 
        self.scale = scale            # image and mask scale
        self.n_channels = n_channels  # input model channels
        self.normalize = normalize    # normalization switch

        # Make sure the scale is between [0, 1]
        assert 0 < scale <= 1, '[ERROR] Scale must be between 0 and 1'

        # Load the reference image into RAM
        #ref_image = Image.open(ref_image_path)
        
        if np.array(ref_image).shape[-1] > n_channels:
            r, g, b, a = ref_image.split()
            ref_image = Image.merge("RGB", (r,g,b))
        
        # Save the reference image into RAM to be used
        self.ref_image = ref_image.copy()

        # Caching the dataset (WARRING : this needs to be used when you have big RAM memory)
        if cached_data:
            logging.info(f'[INFO] Caching the given dataset with {len(self.imgs_dirs)} images and {len(self.masks_dirs)} masks')
            # Turn on the cach flag
            self.cached_dataset = True

            # Preparing the images and masks lists
            self.cache_imgs, self.cache_masks = [], []
            
            # Cache & pre-process the images and the masks (train/val) ready
            for i in trange(len(imgs_dirs)):
                pil_img = Image.open(self.imgs_dirs[i])
                np_img = self.preprocess(pil_img, self.ref_image, self.n_channels, self.scale, self.normalize, mask=False)
                self.cache_imgs.append(np_img)

                if len(self.masks_dirs) == len(self.imgs_dirs):
                    pil_mask = Image.open(self.masks_dirs[i])
                    np_img = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True)
                    self.cache_masks.append(np_img)
        else:
            logging.info(f'[INFO] Dataset with {len(self.imgs_dirs)} images and {len(self.masks_dirs)} masks')
            
    def __len__(self): return len(self.imgs_dirs)

    def delete_cached_dataset(self):
        try:
            del self.cache_imgs[:]
            del self.cache_masks[:]
            logging.info(f'[INFO] All cache deleted')
            return True
        except:
            return False

    def preprocess(self, pil_img, ref_image, n_channels, scale, normalize, mask=True):
        if not(mask):

            if np.array(pil_img).shape[-1] > n_channels:
                r, g, b, a = pil_img.split()
                pil_img = Image.merge("RGB", (r,g,b))

             # This part is concerns the normalization 
            if normalize:
                # Make sure the reference image and the current image have the same size
                assert pil_img.size == ref_image.size, \
                f'Image and reference image should be the same size for histograms matching, but are {pil_img.size} and {ref_image.size}'
           
                if n_channels == 3: pil_img = Image.fromarray(match_histograms(np.array(pil_img),np.array(ref_image), multichannel=True))
                else: pil_img = Image.fromarray(match_histograms(np.array(pil_img),np.array(ref_image)))
            
        # Rescale the image if needed
        if scale != 1 :
            # Get the H and W of the img
            w, h = pil_img.size

            # Get the extimated new size
            newW, newH = int(scale * w), int(scale * h)

            # Resize the image according the given scale
            pil_img = pil_img.resize((newW, newH))

        # Uncomment to convert imgs into gray scale imgs
        # if n_channels != 3: pil_img = pil_img.convert("L")

        # Convert the PIL image into numpy array
        np_img = np.array(pil_img)

        # Add an extra dim if only H, W image
        if len(np_img.shape) == 2: np_img = np.expand_dims(np_img, axis=2)

        # Re-arange the image from (H, W, C) to (C ,H ,W)
        np_img_ready = np_img.transpose((2, 0, 1))
        
        # Ensure the imgs to be in [0, 1]
        if np_img_ready.max() > 1: np_img_ready = np_img_ready / 255
        
        return np_img_ready
    
    def __getitem__(self, i):
        # When the dataset is cached load the img and mask from RAM
        if self.cached_dataset:
            np_img = self.cache_imgs[i]
            if len(self.masks_dirs) == len(self.imgs_dirs):
                np_mask = self.cache_masks[i]
        
        # Otherwise load the img and mask from Disk to RAM
        else:
            # Load the image 
            img_dir = self.imgs_dirs[i]
            pil_img = Image.open(img_dir)

            # Preprocess the image 
            np_img = self.preprocess(pil_img, self.ref_image, self.n_channels, self.scale, self.normalize, mask=False)

            # Load & pre-process the mask if possible
            if len(self.masks_dirs) == len(self.imgs_dirs):
                mask_dir = self.masks_dirs[i]
                pil_mask = Image.open(mask_dir)
                np_mask = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True)
            
            
        if len(self.masks_dirs) == len(self.imgs_dirs):
            return {
                'image': torch.from_numpy(np_img).type(torch.FloatTensor),
                'mask': torch.from_numpy(np_mask).type(torch.FloatTensor)
            }
        else:
            return {
                'image': torch.from_numpy(np_img).type(torch.FloatTensor)
            }

