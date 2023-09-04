from sklearn.preprocessing import LabelEncoder
import numpy as np

np.random.seed(seed)
filter_nc = np.isin(ori_y,np.array(["0", "1","2","3","4","5","6","7","8"]))
filter_target = np.isin(ori_y,np.array([str(target_number)]))
nc_X = ori_X[filter_nc].values
target_X = ori_X[filter_target].values
b_ratio = 0.5
d1,d2 = 28,28
train_flags = np.random.binomial(n=mean_bag_length , p=1/mean_bag_length, size=num_bags_train+num_bags_test)
def generate_bags(num_bags_train,num_bags_test,mean_bag_length,train_flags,nc_X,target_X):

    num_in_train = 60000
    num_in_test = 10000
    train_loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])),
                    batch_size= num_in_train,
                    shuffle=False)
    test_loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                            train=True,
                            download=True,
                            transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size= num_in_test,
                shuffle=False)
    X,Y  = [], []
    num = num_bags_train + num_bags_test
    for i in range(num):
        target_flag = train_flags[i]
        label = 0
        number_of_rows = nc_X.shape[0]
        random_indices = np.random.choice(number_of_rows,size= mean_bag_length - target_flag,replace=False)
        nc_tmp = nc_X[random_indices,:].reshape(-1,d1,d2)
        nc_X = np.delete(nc_X,random_indices,axis = 0)
        if target_flag >0 :
            number_of_rows = target_X.shape[0]
            random_indices = np.random.choice(number_of_rows,size= target_flag,replace=False)
            target_tmp = target_X[random_indices,:].reshape(-1,d1,d2)
            target_X = np.delete(target_X,random_indices,axis = 0)
            nc_tmp = np.vstack((nc_tmp,target_tmp))
            np.random.shuffle(nc_tmp)
            label = 1
        X.append(nc_tmp)
        Y.append(label)
    # Y =  LabelEncoder().fit_transform(Y)
    return X[:num_bags_train],Y[:num_bags_train],X[num_bags_train:],Y[num_bags_train:]

X_train,Y_train,X_test,Y_test = generate_bags(num_bags_train,num_bags_test,mean_bag_length,train_flags,nc_X,target_X)

batch_size  = 1
train_loader, test_loader = convert_data(X_train,Y_train,X_test,Y_test,batch_size= batch_size)