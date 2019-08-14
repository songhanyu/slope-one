import math


# 对从文本读出每行记录进行匹配
def data_filter(user, movie, timestamp, rating):
    return {"user": user, "movie": movie, "rating": rating}
    #


# 生成矩阵(其实是二维字典)
def formMatrix():
    matrix = {}
    f = open("D:/a毕设/data/ml-latest-small/train.txt", "r")
    # f = open("D:/matrix.txt", "r")  # 测试
    # f = open("D:/a毕设/data/1m/train.txt", "r")  # 测试
    lines = f.readlines()
    for line in lines:
        user, movie, timestamp, rating = line.split(' ')
        data = data_filter(user, movie, timestamp, rating)  # 正则匹配训练集中的数据user-movie -rating
        if data["user"] in matrix:
            matrix[data["user"]][data["movie"]] = data["rating"]
        else:  # user不在矩阵中，
            matrix[data["user"]] = {data["movie"]: data["rating"]}
    f.close()
    # print(matrix)
# {'1': {'1': '4', '3': '4', '6': '4'}, '3': {'72378': '0.5'}, '4': {'21': '3', '32': '2'}, '5': {'608': '3'}}
    return matrix


# 计算用户的平均rating
def cal_avg(user_vector):
    user_sum = 0.0
    for i in user_vector:
        user_sum += float(user_vector[i])
    avg = user_sum / len(user_vector)
    return avg


def slopeOne(matrix, test_user, test_movie):
    predict_list = []  # 初始化列表，用来存放一个电影对test_movie 的评分
    movie_list_of_test_user = matrix[test_user]  # 在训练集中找到测试用户test_user看过的电影list
    # print('movie_list_of_test_user', movie_list_of_test_user) # 正确
    # 对于每一部电影，都计算它和test_movie的差值(b)，最终算出相对于它的test_movie的rating
    for movie in movie_list_of_test_user:  # movie是测试用户test_user看过的一个电影
        diff_sum = 0.0
        user_num = 0
        for user in matrix:  # 遍历矩阵，寻找目标用户
            user_movie = matrix[user]  # 找到目标（随便一个）用户看过的电影和评分集合 user_movie
            if test_movie in user_movie and movie in user_movie:  # movie代表其他的电影
                diff_sum += float(user_movie[test_movie]) - float(user_movie[movie])  # 加了float
                user_num += 1
        if user_num:  # 如果找到共同评价过的用户的话
            diff_avg = diff_sum / user_num  # 求出一个其他电影对test_movie的评分差值
            predict_rate = float(movie_list_of_test_user[movie]) + float(diff_avg)  # 测试用户对一个movie的评分 + 偏差; float处理
            predict_list.append((predict_rate, user_num))  # 向列表list末尾加数据
            print('新一个预测的分数：', predict_rate)

    # 如果没人看过，取这个人的平均分
    if not predict_list:
        avg = cal_avg(movie_list_of_test_user)
        # print avg
        return avg

    # 算出它的rating(原生Slope One)
    molecule = 0.0  # molecule 分子 molecular 分子的
    denominator = 0.0  # dɪˈnɒmɪneɪtə(r) 分母
    for predict in predict_list:
        molecule += predict[0]  # 原生Slope One
    denominator = len(predict_list)
    # print('predict_list:', predict_list)
    # print('predict_list长度:', denominator)
    result = molecule / denominator
    print('预测分：', result)
    return molecule / denominator


# 加权Slope One
def weightSlopeOne(matrix, test_user, test_movie):
    predict_list = []  # 初始化列表，用来存放一个电影对test_movie 的评分
    movie_list_of_test_user = matrix[test_user]  # 在训练集中找到测试用户test_user看过的电影list
    # print('movie_list_of_test_user', movie_list_of_test_user) # 正确
    # 对于每一部电影，都计算它和test_movie的差值(b)，最终算出相对于它的test_movie的rating
    for movie in movie_list_of_test_user:  # movie是测试用户test_user看过的一个电影
        diff_sum = 0.0
        user_num = 0
        for user in matrix:  # 遍历矩阵，寻找目标用户
            user_movie = matrix[user]  # 找到目标用户看过的电影和评分集合 user_movie
            if test_movie in user_movie and movie in user_movie:  # movie代表其他的电影
                diff_sum += float(user_movie[test_movie]) - float(user_movie[movie])  # 加了float
                user_num += 1
        if user_num:  # 如果找到共同评价过的用户的话
            diff_avg = diff_sum / user_num  # 求出一个其他电影对test_movie的评分差值
            predict_rate = float(movie_list_of_test_user[movie]) + float(diff_avg) # 测试用户对一个movie的评分 + 偏差; float处理
            predict_list.append((predict_rate, user_num))  # 向列表list末尾加数据
    # 如果没人看过，取这个人的平均分
    if not predict_list:
        avg = cal_avg(movie_list_of_test_user)
        return avg
    # 算出它的rating(原生Slope One)
    molecule = 0.0  # molecule 分子 molecular 分子的
    denominator = 0.0  # dɪˈnɒmɪneɪtə(r) 分母
    for predict in predict_list:
        molecule += predict[0] * predict[1]  # 加权Slope One
        denominator += predict[1]  # 共同评价用户个数
    print('predict_list:', predict_list)
    print('predict_list长度:', denominator)
    print('分子是：', molecule)
    print('分母是：', denominator)
    # return int(round(molecusar / denominator))
    result = molecule / denominator
    print('预测分：', result)
    return molecule / denominator


# 融合用户相似度：
def userbased_cal_matrix(matrix, test_user, test_movie):
    # STEP 1 计算test_user的均值
    test_user_vector = matrix[test_user]
    test_user_avg = cal_avg(test_user_vector)  # test_user的评分均值
    # print test_user_vector
    # print test_user_avg
    # 只选为test_movie电影打过分的user
    sim_list = []
    for user in matrix:
        if test_movie in matrix[user]:
            user_vector = matrix[user]
            # user_vector 得到 user train用户平价过的电影及其评分
            # user_avg 用户评价所有电影的均值
            user_avg = cal_avg(user_vector)
            molecusar = 0.0  # 分子
            denominatorA = 0.0  # 分母
            denominatorB = 0.0
            for key in test_user_vector:  # test_user_vector=matrix[test_user]
                if key in user_vector:  # user_vector = matrix[user]
                    a = float(test_user_vector[key]) - test_user_avg
                    b = float(user_vector[key]) - user_avg  # user_avg训练集中的用户评分均值
                    molecusar += a * b
                    denominatorA += a * a
                    denominatorB += b * b
            if denominatorA and denominatorB and molecusar:
                # sim 是皮尔逊系数计算的用户相似度
                sim = molecusar / math.sqrt(denominatorA) / math.sqrt(denominatorB)
                print(user, '的相似度是：', sim)
                #           用户     用户评价均值    用户评价的test_movie的评分  用户相似度
                # sim_list.append((user, user_avg, user_vector[test_movie], abs(sim)))
                # if sim > 0:
                #     sim_list.append((user, user_vector[test_movie], sim))
                sim_list.append((user, user_vector[test_movie], abs(sim)))

    # print sim_list

    molecusar = 0.0
    denominator = 0.0
    if sim_list:
        for data in sim_list:
            u_vector = matrix[data[0]]  # 训练集中的电影评分集合
            for mov in test_user_vector:  # 测试用户的mov
                if mov in u_vector:
                    dev = float(data[1]) - float(u_vector[mov])
                    # print('dev是：', dev)
                    molecusar += (float(test_user_vector[mov]) + dev) * data[2]
                    denominator += data[2]

            # 分子：（某用户对test_movie的评分 - 该用户的评分均值）* 皮尔逊相似度
            # test_user_vector = matrix[test_user]
            # molecusar += data[3] * (float(data[2]) - data[1])
            # 此处改动：dev i,j = data[2] - data[1]
            # molecusar += data[3] * (float(data[2]) - data[1])
            # denominator += abs(data[3])
            # print(data[0], ' 评分是：', data[2])
            # print(data[0], ' 与测试用户相似度是：', abs(data[3]))
        # 测试用户的平均评分 +
        pearson_rating = molecusar / denominator
    else:
        # 如果没有相似集（本质上是因为没人看过这电影），取自己打分的平均值
        pearson_rating = test_user_avg
    print('最终评分：', pearson_rating)
    return pearson_rating


def hahah():
    matrix = formMatrix()
    f = open("D:/test.txt", "r")
    lines = f.readlines()
    for line in lines:
        test_user, test_movie, test_timestamp, test_rating = line.split(' ')
        print('hhh_test_user；', test_user)
        print('jjj_test_movie；', test_movie)
        print('hhh_test_rating；', test_rating)
        print('hhh_test_timestamp；', test_timestamp)
    userbased_cal_matrix(matrix, str(test_user), str(test_movie))


# hahah()


def test():
    mae_sum = 0
    rmae_sum = 0
    number = 0

    matrix = formMatrix()
    f = open("D:/a毕设/data/ml-latest-small/test.txt", "r")
    # wf = open("D:/output.txt", 'w') # 原生Slope One
    # wf = open("D:/output2.txt", 'w')  # 加权Slope One
    # f = open("D:/a毕设/data/1m/test.txt", "r")
    # wf = open("D:/output3.txt", 'w')  # 用户相似度 Slope One
    lines = f.readlines()
    for line in lines:
        test_user, test_movie, test_timestamp, test_rating = line.split(' ')  # 改动
        # pre_rating = slopeOne(matrix, test_user, test_movie)
        pre_rating = weightSlopeOne(matrix, test_user, test_movie)  # 加权Slope One
        # pre_rating = userbased_cal_matrix(matrix, test_user, test_movie)  # 融合用户相似度 Slope One
        print('预测分数：', pre_rating)
        # string = str(test_user) + "\t" + str(test_movie) + "\t" + str(pre_rating) + "\t" + str(test_rating) + "\n"
        # wf.write(string)
        t = pre_rating - float(test_rating)
        print('pre_rating:', pre_rating)
        print('test_rating:', test_rating)
        # print('t是：', t)
        mae_sum += abs(t)
        rmae_sum += t * t
        number += 1
    # wf.close()
    print("MAE: " + str(mae_sum * 1.0 / number))
    print("RMAE: " + str(math.sqrt(rmae_sum * 1.0 / number)))


test()

# 原生Slope One：
# MAE: 0.6769131087090916
# RMAE: 0.8843319926047358

# 加权 Slope One：
# MAE: 0.6713514812459522
# RMSE: 0.8759307095258512

# 融合用户相似度Slope One：
# MAE: 0.6696207544508279
# RMAE: 0.8736040709515789
#
# MAE: 0.6676222236025383
# RMSE: 0.8714649227921208
#
# 去掉小于0的相似度在计算的结果：
# MAE: 0.6645386959053933
# RMSE: 0.8677811484465896

