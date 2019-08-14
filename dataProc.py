import sys
import random
random.seed(0)


class DataProc():
    def __init__(self):  # 构造函数；self代表this指针
        self.trainset = {}
        self.testset = {}


# load的文件是rating.dat
    @staticmethod
    def loadfile(filename):
        """
        :param filename: load a file,
        :return: a generator
        """
        print("loadfile filename =", filename)
        fp = open(filename, 'r')
        # line代表数据集重每行的内容，i表示当前读到第几行，每当读到100000行的整数倍时输出提示语句
        # enumerate用于将一个可遍历的数据对象组合为一个索引序列
        for i, line in enumerate(fp):
            # 删除每行的回车符的ASCII编码，yield加强版的return，可以返回多个元素
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print(sys.stderr, 'load %s(%s)' % (filename, i))
        fp.close()
        print(sys.stderr, 'load %s success' % filename)

    def generate_dataset(self, filename, pivot=0.8):
        # load rating data and split it to training set and test set
        trainset_len = 0
        testset_len = 0
        train_file = "D:\\a毕设\\data\\ml-latest-small\\train.txt"
        output1 = open(train_file, 'w')
        test_file = "D:\\a毕设\\data\\ml-latest-small\\test.txt"
        output2 = open(test_file, 'w')
        for line in self.loadfile(filename):  # self.loadfile
            user, movie, rating, timestamp = line.split('::')
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = rating
                trainset_len += 1
                # 加%d是为了让数字转换成字符串     此处增加时间戳
                train_str = str(user) + ' ' + str(movie) + ' ' + str(timestamp) + ' ' + self.trainset[user][
                    movie] + '\n'
                output1.write(train_str)
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = rating
                testset_len += 1
                test_str = str(user) + ' ' + str(movie) + ' ' + str(timestamp) + ' ' + self.testset[user][
                    movie] + '\n'
                output2.write(test_str)

        output1.close()
        output2.close()
        print(sys.stderr, 'split train set and test set success')
        print(sys.stderr, 'train set = %s' % trainset_len)
        print(sys.stderr, 'test set = %s' % testset_len)


# if _name_ == '_main_' 两个_ _
# 在一个函数或一个类之后，要空两行
if __name__ == '__main__':
    ratingfile = 'D:/a毕设/data/ml-latest-small/ratings.csv'  # 实验使用的是ml-latest-small数据集
    datproc = DataProc()
    datproc.generate_dataset(ratingfile)





