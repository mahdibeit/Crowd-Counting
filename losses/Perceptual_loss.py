#This part should be included in vgg.py
class AE(nn.Module):
      def __init__(self):
            super(AE, self).__init__()
            self.encoder = nn.Sequential(
                  nn.Conv2d(1, 64, 3, stride=2,padding=1),
                  nn.ReLU(True),
                  nn.Conv2d(64, 128, 3, stride=2,padding=1),
                  nn.ReLU(True),
                  nn.Conv2d(128, 16, 3, stride=2,padding=1),
                  nn.ReLU(True),
                  nn.Conv2d(16, 3, 3, stride=2,padding=1),
                  nn.ReLU(True))
                  

            self.decoder = nn.Sequential(
            
                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(3, 16, 3, stride=1,padding=1),
                  nn.ReLU(True),
                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(16, 128, 3, stride=1,padding=1),
                  nn.ReLU(True),
                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(128, 64, 3, stride=1,padding=1),
                  nn.ReLU(True),
                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(64, 1, 3, stride=1,padding=1),                  
                  nn.Sigmoid())

      def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x


def Feature_extractor():
    model = AE()
    model.load_state_dict(torch.load(r"C:\Users\mahdi\Desktop\SASNet_ROOT\Bayseian\AE\model.pth"), strict=False)
    #Freeze the feature extractor
    for param in model.parameters():
        param.requires_grad = False
    encoder = model.encoder
    return encoder


#This customized training function should be included in regression_trainer.py

class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        self.tb_writer = SummaryWriter(r"C:\Users\mahdi\Desktop\SASNet_ROOT\Bayseian\logs/{}".format(time.time()),flush_secs=1)
        print("tensorboard is running")
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")
        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model =vgg19()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.feature_extractor = Feature_extractor()
        self.feature_extractor.to(self.device)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        self.perceptual_loss = torch.nn.MSELoss()
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def keypoint2densitymap(self, keypoints, img_size, pm_size):  
        keypoints = keypoints[0].cpu().detach().numpy()
        keypoints = keypoints

        def gaussian_filter_density(gt):
            density = np.zeros(gt.shape, dtype=np.float32)
            gt_count = np.count_nonzero(gt)
            if gt_count == 0:
                return density

            pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
            leafsize = 2048
            # build kdtree
            tree = KDTree(pts.copy(), leafsize=leafsize)
            # query kdtree
            distances, locations = tree.query(pts, k=4)

            for i, pt in enumerate(pts):
                pt2d = np.zeros(gt.shape, dtype=np.float32)
                pt2d[pt[1],pt[0]] = 1.
                if gt_count > 1:
                    sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
                else:
                    sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
                density += scipy.ndimage.filters.gaussian_filter(pt2d, 2, mode='constant')
            return density

        gt_pm = np.zeros((int(img_size[0]/8),int(img_size[1]/8)))
        gt_pm[np.clip((keypoints[:,1]/8).astype(np.int),0,int(img_size[0]/8-1)),np.clip((keypoints[:,0]/8).astype(np.int),0,int(img_size[1]/8-1))]=1
        gt_dm = gaussian_filter_density(gt_pm)
        gt_dm = cv2.resize(gt_dm, (pm_size[1],pm_size[0]), interpolation=cv2.INTER_CUBIC)
        # gt_dm_256 = (gt_dm*255).astype(np.uint8)
        # # plt.imshow(gt_dm)
        # # plt.savefig('gt_dm.png')
        # #Apply CLAHE using cv2 library
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # img = clahe.apply(gt_dm_256)
        # # plt.imshow(img)
        # # plt.savefig('img.png')
        # img = img*np.sum(gt_dm)/np.sum(img)
        # img = img/255.
        # img = img.astype(gt_dm.dtype)
        return gt_dm

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                prob_list = self.post_prob(points, st_sizes)
                img_size = (inputs.size()[-2],inputs.size()[-1])
                output_size = (outputs.size()[-2],outputs.size()[-1])
                GT_density_map = self.keypoint2densitymap(points, img_size, output_size)
                GT_density_map = GT_density_map.reshape(1,1,GT_density_map.shape[0],GT_density_map.shape[1])
                GT_density_map = torch.from_numpy(GT_density_map).to(self.device)
                # GT_density_map = GT_density_map.double()
                GT_density_map_features = self.feature_extractor(GT_density_map)
                output_features = self.feature_extractor(outputs)
                # loss = self.criterion(prob_list, targets, outputs) + self.perceptual_loss(output_features, GT_density_map_features) #Combined loss
                loss = self.perceptual_loss(output_features, GT_density_map_features) #Perceptual loss


                # loss = self.perceptual_loss(outputs, GT_density_map) #MSE loss
                # loss = self.criterion(prob_list, targets, outputs) #Bayesian loss

                # #Channelwise MSE of two pytorch tensors
                # MSE_map = (output_features - GT_density_map_features)**2
                # MSE_map_upsampled = F.interpolate(MSE_map, size=(GT_density_map.size()[-2],GT_density_map.size()[-1]), mode='bicubic', align_corners=True)
                # Density_MSE = (GT_density_map - outputs)**2
                # Weighted_MSE = Density_MSE * MSE_map_upsampled
                # loss = 0.4*torch.mean(Weighted_MSE) + 0.2*self.perceptual_loss(outputs, GT_density_map) + 0.4*self.perceptual_loss(output_features, GT_density_map_features)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time()-epoch_start))

        self.tb_writer.add_scalar('train_loss', epoch_loss.get_avg(), self.epoch)
        self.tb_writer.add_scalar('train_MSE', np.sqrt(epoch_mse.get_avg()), self.epoch)
        self.tb_writer.add_scalar('train_MAE', epoch_mae.get_avg(), self.epoch)


        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
        #Copnvert model output to numpy array
        output_np = outputs.detach().cpu().numpy()
        plt.imshow(output_np[0,0,:,:])
        plt.imsave(r"C:\Users\mahdi\Desktop\SASNet_ROOT\Bayseian\Bayesian-Crowd-Counting-master\smaple%d.png"%(np.random.randint(100)),output_np[0,0,:,:])
        
        print("$$$$$$$$$$$$$$$$$$$$image saved$$$$$$$$$$$$$$$$$$$$$$$$$$")
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        self.tb_writer.add_scalar('val_MSE', mse, self.epoch)
        self.tb_writer.add_scalar('val_MAE', mae, self.epoch)

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))


