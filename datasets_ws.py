# import torchvision; torchvision.utils.save_image(views[0].float(), 'debug_output_views.png')
import os
import cv2
import torch
import faiss
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
from scipy.io import loadmat
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader

from scipy.spatial.distance import pdist
import math
from util import Raw2Celsius

base_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collate_fn(batch):
    images  = torch.cat([e[0][None] for e in batch])
    pos_view  = torch.cat([e[1][None] for e in batch])
    local_views = torch.cat([e[2][None] for e in batch])
    return images, pos_view, local_views

class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.split = split

        # 1. Custom Dataset Sequence 확인 (ms2 vs sthereo)
        if all(i in ['Campus', 'Residential', 'Urban'] for i in args.sequences):
            self.dataset_type = 'ms2'
        elif all(i in ['KAIST', 'SNU', 'Valley'] for i in args.sequences):
            self.dataset_type = 'sthereo'
        elif all(i in ['r0', 'r1'] for i in args.sequences):
            self.dataset_type = 'nsavp'
        else:
            raise Exception("sequence typo i guess")
            
        self.img_time = args.img_time
        
        # 2. matStruct 로드
        self.matStruct = [
            loadmat(os.path.join(datasets_folder, "save_mat", split, seq, f'{self.dataset_type}_{split}.mat'))['dbStruct']
            for seq in args.sequences
        ]
        
        self.resize = args.resize
        self.test_method = args.test_method
        
        # 3. Database & Queries UTM 좌표 확보
        self.database_utms = np.concatenate([mat['db_pose'][0, 0] for mat in self.matStruct])
        
        if self.dataset_type in ['ms2', 'sthereo']:
            if self.img_time == 'allday':
                self.queries_utms = np.concatenate([np.concatenate((mat['q_pose_morning'][0, 0], mat['q_pose_afternoon'][0, 0], mat['q_pose_evening'][0, 0])) if self.dataset_type == 'sthereo' else np.concatenate((mat['q_pose_morning'][0, 0], mat['q_pose_clearsky'][0, 0], mat['q_pose_rainy'][0, 0], mat['q_pose_nighttime'][0, 0])) for mat in self.matStruct])
            elif self.img_time == 'daytime':
                self.queries_utms = np.concatenate([np.concatenate((mat['q_pose_morning'][0, 0], mat['q_pose_afternoon'][0, 0])) if self.dataset_type == 'sthereo' else np.concatenate((mat['q_pose_morning'][0, 0], mat['q_pose_clearsky'][0, 0], mat['q_pose_rainy'][0, 0])) for mat in self.matStruct])
            elif self.img_time == 'nighttime':
                self.queries_utms = np.concatenate([mat['q_pose_evening'][0, 0] if self.dataset_type == 'sthereo' else mat['q_pose_nighttime'][0, 0] for mat in self.matStruct])
            elif self.img_time == 'latetime':
                self.queries_utms = np.concatenate([np.concatenate((mat['q_pose_afternoon'][0, 0], mat['q_pose_evening'][0, 0])) if self.dataset_type == 'sthereo' else mat['q_pose_nighttime'][0, 0] for mat in self.matStruct])
        elif self.dataset_type in ['nsavp']:
            if 'r0' in self.args.sequences:
                self.queries_utms = np.concatenate(
                    [np.concatenate((mat['q_pose_FA0'][0, 0], mat['q_pose_FN0'][0, 0], mat['q_pose_FS0'][0, 0])) for mat in self.matStruct])
            elif 'r1' in self.args.sequences:
                self.queries_utms = np.concatenate(
                    [np.concatenate((mat['q_pose_FA0'][0, 0], mat['q_pose_DA0'][0, 0])) for mat in self.matStruct])
            else:
                print(self.dataset_type)
                raise Exception("What?")
            
        # 4. Soft Positives 계산 (기존 BaseDataset 로직 유지)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        if split == "train":
            self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms, radius=25, return_distance=False)
        else:
            self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms, radius=args.val_positive_dist_threshold, return_distance=False)            
        
        # 5. Database(RGB) & Queries(Thermal) 경로 확보
        self.database_paths = np.concatenate([mat['db_rgb'][0, 0] for mat in self.matStruct])
        
        if self.dataset_type in ['ms2', 'sthereo']:
            if self.img_time == 'allday':
                self.queries_paths = np.concatenate([np.concatenate((mat['q_t_morning'][0, 0], mat['q_t_afternoon'][0, 0], mat['q_t_evening'][0, 0])) if self.dataset_type == 'sthereo' else np.concatenate((mat['q_t_morning'][0, 0], mat['q_t_clearsky'][0, 0], mat['q_t_rainy'][0, 0], mat['q_t_nighttime'][0, 0])) for mat in self.matStruct])
            elif self.img_time == 'daytime':
                self.queries_paths = np.concatenate([np.concatenate((mat['q_t_morning'][0, 0], mat['q_t_afternoon'][0, 0])) if self.dataset_type == 'sthereo' else np.concatenate((mat['q_t_morning'][0, 0], mat['q_t_clearsky'][0, 0], mat['q_t_rainy'][0, 0])) for mat in self.matStruct])
            elif self.img_time == 'nighttime':
                self.queries_paths = np.concatenate([mat['q_t_evening'][0, 0] if self.dataset_type == 'sthereo' else mat['q_t_nighttime'][0, 0] for mat in self.matStruct])
            elif self.img_time == 'latetime':
                self.queries_paths = np.concatenate([np.concatenate((mat['q_t_afternoon'][0, 0], mat['q_t_evening'][0, 0])) if self.dataset_type == 'sthereo' else mat['q_t_nighttime'][0, 0] for mat in self.matStruct])
        elif self.dataset_type in ['nsavp']:
            if 'r0' in self.args.sequences:
                self.queries_paths = np.concatenate(
                    [np.concatenate((mat['q_t_FA0'][0, 0], mat['q_t_FN0'][0, 0], mat['q_t_FS0'][0, 0])) for mat in self.matStruct])
            elif 'r1' in self.args.sequences:
                self.queries_paths = np.concatenate(
                    [np.concatenate((mat['q_t_FA0'][0, 0], mat['q_t_DA0'][0, 0])) for mat in self.matStruct])
            else: raise Exception("What?")
        

        # 통합된 경로 리스트 및 개수 정의
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        self.database_num = len(self.database_paths)
        self.queries_num  = len(self.queries_paths)
        if self.dataset_type=='ms2':
            self.min_temp, self.max_temp = -20, 60

    # 6. OpenCV로 읽은 이미지를 PIL 포맷으로 변환 (기존 BaseDataset과의 호환성을 위해)
    def get_rgb_img(self, path):
        path = str(path).strip("[]'") # 매트랩 배열 찌꺼기 방지
        if self.dataset_type == 'sthereo':
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
        elif self.dataset_type == 'ms2':  # ms2
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.dataset_type == 'nsavp':
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = img[196:820, 230:1010]
        else: raise Exception("ERROR datasets_ws.py :: get_rgb_img")
        
        return Image.fromarray(img)

    def get_thermal_img(self, path):
        path = str(path).strip("[]'")
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if self.dataset_type == 'ms2':
            img = Raw2Celsius(img)
            img = np.clip(img, self.min_temp, self.max_temp)
        elif self.dataset_type == 'nsavp':
            img = np.clip(img, 22500, 25000).astype(np.float32)                                                             
            img = (img - 22500) / 2500.0 * 255.0
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img)

    def __getitem__(self, index):
        # Database 인덱스면 RGB, Query 인덱스면 Thermal 호출
        if index < self.database_num:
            img = self.get_rgb_img(self.images_paths[index])
            img = base_transform(img)
        else:
            img = self.get_thermal_img(self.images_paths[index])
            img = base_transform(img)
            
        if self.test_method == "hard_resize":
            img = transforms.functional.resize(img, self.resize)
        else:
            img = self._test_query_transform(img)
            
        return img, index
    
    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "single_query":
            processed_img = transforms.functional.resize(img, min(self.resize))
        elif self.test_method == "central_crop":
            scale = max(self.resize[0]/H, self.resize[1]/W)
            processed_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            processed_img = transforms.functional.center_crop(processed_img, self.resize)
        elif self.test_method in ["five_crops", "nearest_crop", "maj_voting"]:
            shorter_side = min(self.resize)
            processed_img = transforms.functional.resize(img, shorter_side)
            processed_img = torch.stack(transforms.functional.five_crop(processed_img, shorter_side))
        return processed_img
    
    def __len__(self):
        return len(self.images_paths)
        
    def __repr__(self):
        return (f"< {self.__class__.__name__}, {self.dataset_name} ({self.dataset_type}) - "
                f"#database(RGB): {self.database_num}; #queries(Thermal): {self.queries_num} >")
        
    def get_positives(self):
        return self.soft_positives_per_query

    def get_metadata(self):
        """각 이미지(DB + queries)에 대한 (modality, sequence, condition) 메타데이터 반환."""
        modalities, seq_labels, conditions = [], [], []

        # Database images: RGB, condition='database'
        for mat, seq in zip(self.matStruct, self.args.sequences):
            n = len(mat['db_rgb'][0, 0])
            modalities  += ['RGB']      * n
            seq_labels  += [seq]        * n
            conditions  += ['database'] * n

        # Query images: Thermal, condition=time condition
        if self.dataset_type == 'sthereo':
            if self.img_time == 'allday':
                time_conds = ['morning', 'afternoon', 'evening']
            elif self.img_time == 'daytime':
                time_conds = ['morning', 'afternoon']
            elif self.img_time == 'nighttime':
                time_conds = ['evening']
            elif self.img_time == 'latetime':
                time_conds = ['afternoon', 'evening']
        elif self.dataset_type == 'ms2':
            if self.img_time == 'allday':
                time_conds = ['morning', 'clearsky', 'rainy', 'nighttime']
            elif self.img_time == 'daytime':
                time_conds = ['morning', 'clearsky', 'rainy']
            elif self.img_time == 'nighttime':
                time_conds = ['nighttime']
            elif self.img_time == 'latetime':
                time_conds = ['nighttime']
        elif self.dataset_type == 'nsavp':
            if 'r0' in self.args.sequences:
                time_conds = ["FA0", "FN0", "FS0"]
            elif 'r1' in self.args.sequences:
                time_conds = ["FA0", "DA0"]
            else: raise Exception("What?")
                

        for mat, seq in zip(self.matStruct, self.args.sequences):
            for cond in time_conds:
                n = len(mat[f'q_t_{cond}'][0, 0])
                modalities += ['Thermal'] * n
                seq_labels += [seq]       * n
                conditions += [cond]      * n

        return np.array(modalities), np.array(seq_labels), np.array(conditions)

class RAMEfficient2DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]
    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)
    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]

class RAMEfficient4DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 3D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]
    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        assert vals.shape[2] == self.shape[2], f"{vals.shape[2]} {self.shape[2]}"
        assert vals.shape[3] == self.shape[3], f"{vals.shape[3]} {self.shape[3]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)
    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]

class LeJEPADataset(BaseDataset):
    """Dataset used for training, it is used to compute the triplets 
    with TripletsDataset.compute_triplets() with various mining methods.
    If is_inference == True, uses methods of the parent class BaseDataset,
    this is used for example when computing the cache, because we compute features
    of each image, not triplets.
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train", negs_num_per_query=10):
        super().__init__(args, datasets_folder, dataset_name, split)
        self.mining = args.mining
        self.neg_samples_num = args.neg_samples_num  # Number of negatives to randomly sample
        self.negs_num_per_query = negs_num_per_query  # Number of negatives per query in each batch
        if self.mining == "full":  # "Full database mining" keeps a cache with last used negatives
            self.neg_cache = [np.empty((0,), dtype=np.int32) for _ in range(self.queries_num)]
        self.is_inference = False
        
        identity_transform = transforms.Lambda(lambda x: x)
        self.resized_transform = transforms.Compose([
            transforms.Resize(self.resize) if self.resize is not None else identity_transform,
            base_transform
        ])
        
        self.query_transform = transforms.Compose([
            transforms.ColorJitter(contrast=(0.5, 1.0)) \
                if args.contrast != None else identity_transform,
            transforms.RandomResizedCrop(size=self.resize, scale=(1-args.random_resized_crop, 1)) \
                if args.random_resized_crop != None else identity_transform,
            #######################################################
            self.resized_transform,
        ])

        # self.views_transform = transforms.Compose([
        #     transforms.ColorJitter(contrast=(0.5, 1.0)) \
        #         if args.contrast != None else identity_transform,
        #     transforms.RandomResizedCrop(size=self.resize, scale=(0.3, 0.8)) \
        #         if args.random_resized_crop != None else identity_transform,
        #     #######################################################
        #     self.resized_transform,
        # ])
        
        # Find hard_positives_per_query, which are within train_positives_dist_threshold (10 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = list(knn.radius_neighbors(self.queries_utms,
                                             radius=args.train_positives_dist_threshold,  # 10 meters
                                             return_distance=False))
        
        #### Some queries might have no positive, we should remove those queries.
        queries_without_any_hard_positive = np.where(np.array([len(p) for p in self.hard_positives_per_query], dtype=object) == 0)[0]
        if len(queries_without_any_hard_positive) != 0:
            logging.info(f"There are {len(queries_without_any_hard_positive)} queries without any positives " +
                         "within the training set. They won't be considered as they're useless for training.")
        
        # Remove queries without positives
        # self.hard_positives_per_query = np.delete(self.hard_positives_per_query, queries_without_any_hard_positive)
        self.hard_positives_per_query = [
            positives for i, positives in enumerate(self.hard_positives_per_query)
            if i not in queries_without_any_hard_positive
        ]
        self.queries_paths            = np.delete(self.queries_paths,            queries_without_any_hard_positive)
        if len(queries_without_any_hard_positive) != 0:
                    self.soft_positives_per_query = np.delete(self.soft_positives_per_query, queries_without_any_hard_positive, axis=0)
                    self.queries_utms = np.delete(self.queries_utms, queries_without_any_hard_positive, axis=0)        
                    
        # Recompute images_paths and queries_num because some queries might have been removed
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        self.queries_num = len(self.queries_paths)

    def __getitem__(self, index):
        # =========================================================
        # 1. 고정된 역할: Query는 무조건 IR, Positive는 무조건 RGB
        # =========================================================
        if self.is_inference:
            return super().__getitem__(index)
        
        query_index, best_positive_index, _ = torch.split(self.triplets_global_indexes[index], (1,1,self.negs_num_per_query))
        query_img = self.get_thermal_img(self.queries_paths[query_index.item()])
        pos_img = self.get_rgb_img(self.database_paths[best_positive_index.item()])

        # 증강(Transform) 적용
        if self.query_transform: 
            views = [self.query_transform(query_img) for _ in range(2)]
            query_img = views[0]
            view = views[1]
        else: 
            query_img = self.resized_transform(query_img)
        pos_img = self.resized_transform(pos_img)
        
        return query_img, pos_img, view
    
    def __len__(self):
        if self.is_inference:
            # At inference time return the number of images. This is used for caching or computing NetVLAD's clusters
            return super().__len__()
        else:
            return len(self.triplets_global_indexes)
        
    def compute_triplets(self, args, model):
        self.is_inference = True
        if self.mining == "full":
            self.compute_triplets_full(args, model)
        elif self.mining == "partial" or self.mining == "msls_weighted":
            self.compute_triplets_partial(args, model)
        elif self.mining == "random":
            self.compute_triplets_random(args, model)
        else: raise Exception("?")
  
    @staticmethod
    def compute_cache(args, model, subset_ds, cache_shape):
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives."""
        # featurea = []
        # def hook(module, input, output):
        #     featurea.append(output.clone().detach())

        subset_dl = DataLoader(dataset=subset_ds, num_workers=args.num_workers, 
                               batch_size=args.infer_batch_size, shuffle=False,
                               pin_memory=(args.device=="cuda"))
        model = model.eval()
        # RAMEfficient2DMatrix can be replaced by np.zeros, but using
        # RAMEfficient2DMatrix is RAM efficient for full database mining.
        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)
        database_num = subset_ds.dataset.database_num
        with torch.no_grad():
            for images, indexes in tqdm(subset_dl, ncols=100):
                images = images.to(args.device)
                is_thermal = (indexes >= database_num).to(args.device)
                global_features = model(images, is_thermal)
                cache[indexes.numpy()] = global_features.cpu().numpy()
        return cache, None

    def get_query_features(self, query_index, cache):
        query_features = cache[query_index + self.database_num]
        if query_features is None:
            raise RuntimeError(f"For query {self.queries_paths[query_index]} " +
                               f"with index {query_index} features have not been computed!\n" +
                               "There might be some bug with caching")
        return query_features
    
    def get_best_positive_index(self, args, query_index, cache, query_features):
        positives_features = cache[self.hard_positives_per_query[query_index]]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features)
        # Search the best positive (within 10 meters AND nearest in features space)
        _, best_positive_num = faiss_index.search(query_features.reshape(1, -1), 1)
        best_positive_index = self.hard_positives_per_query[query_index][best_positive_num[0]].item()
        return best_positive_index
    
    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_features = cache[neg_samples]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features)
        # Search the 10 nearest negatives (further than 25 meters and nearest in features space)
        _, neg_nums = faiss_index.search(query_features.reshape(1, -1), self.negs_num_per_query)
        neg_nums = neg_nums.reshape(-1)
        neg_indexes = neg_samples[neg_nums].astype(np.int32)
        return neg_indexes
    
    def compute_triplets_random(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)
        # Take all the positives
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]  # Flatten list of lists to a list
        positives_indexes = list(np.unique(positives_indexes))
        
        # Compute the cache only for queries and their positives, in order to find the best positive
        subset_ds = Subset(self, positives_indexes + list(sampled_queries_indexes + self.database_num))
        cache, _ = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            
            # Choose some random database images, from those remove the soft_positives, and then take the first 10 images as neg_indexes
            soft_positives = self.soft_positives_per_quer[query_index]
            neg_indexes = np.random.choice(self.database_num, size=self.negs_num_per_query+len(soft_positives), replace=False)
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)[:self.negs_num_per_query]
            
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)
    
    def compute_triplets_full(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)
        # Take all database indexes
        database_indexes = list(range(self.database_num))
        #  Compute features for all images and store them in cache
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        cache, _ = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            # Choose 1000 random database images (neg_indexes)
            neg_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False)
            # Remove the eventual soft_positives from neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)
            # Concatenate neg_indexes with the previous top 10 negatives (neg_cache)
            neg_indexes = np.unique(np.concatenate([self.neg_cache[query_index], neg_indexes]))
            # Search the hardest negatives
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            # Update nearest negatives in neg_cache
            self.neg_cache[query_index] = neg_indexes
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)
    
    def compute_triplets_partial(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        if self.mining == "partial":
            sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)
        
        # Sample 1000 random database images for the negatives
        neg_sample_size = min(self.neg_samples_num, self.database_num)                                                                                                           
        sampled_database_indexes = np.random.choice(self.database_num, neg_sample_size, replace=False)
        # Take all the positives
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]
        # Merge them into database_indexes and remove duplicates
        database_indexes = list(sampled_database_indexes) + positives_indexes
        database_indexes = list(np.unique(database_indexes))
        
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        cache, _ = self.compute_cache(args, model, subset_ds, cache_shape=(len(self), args.features_dim))
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            
            # Choose the hardest negatives within sampled_database_indexes, ensuring that there are no positives
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
            
            # Take all database images that are negatives and are within the sampled database images (aka database_indexes)
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)
