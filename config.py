#coding:utf-8
import warnings

class DefaultConfig(object):
    env = 'default' # visdom 环境
    model = 'FaceNet' # 使用的模型，名字必须与models/__init__.py中的名字一致

    data_rootdir = '/media/haitao/zxy/ECUST2019'
    
    # 训练集存放路径
    train_data_root = '/home/xianyi/Data/vggface2/train_mtcnn_182' #　vggface数据,没有检测
    # 验证集存放路径
    valid_data_root = '/home/xianyi/Data/lfw/lfw_mtcnn_182'
    # 测试集存放路径
    test_data_root = '/media/haitao/zxy/Data/lfw/lfw_mtcnn_182'
 
    train_split = '/media/haitao/zxy/new_split/split1/train.txt'
    valid_split = '/media/haitao/zxy/new_split/split1/valid.txt'
    test_split= '/media/haitao/zxy/new_split/split1/test.txt'

    # 加载预训练的模型的路径，为None代表不加载
    load_model_path = 'checkpoints/facenet_0402_104030.pth' #'checkpoints/model.pth' 
   
    batch_size = 32 # batch size 一次性读取多少数据，内存不够的话少一点
    num_workers = 8 # how many workers for loading data
    print_freq = 20 # print info every N batch
    valid_freq = 1 # N epoch 之后验证
    test_freq =1

    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
      
    max_epoch = 100
    lr = 0.001# initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数
    step_size = 50 # 学习率自动调节中的更新学习率的步长

    num_classes = 10000 # 需要分的类
    
    netinputsize = 7 # 网络输入图片的大小


# def parse(self,kwargs):
#         '''
#         根据字典kwargs 更新 config参数
#         '''
#         for k,v in kwargs.iteritems():
#             if not hasattr(self,k):
#                 warnings.warn("Warning: opt has not attribut %s" %k)
#             setattr(self,k,v)

#         print('user config:')
#         for k,v in self.__class__.__dict__.iteritems():
#             if not k.startswith('__'):
#                 print(k,getattr(self,k))


# DefaultConfig.parse = parse
opt =DefaultConfig()
# opt.parse = parse
