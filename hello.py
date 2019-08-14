import math


# 对从文本读出每行记录进行匹配
def data_filter(user, movie, timestamp, rating):
    return {"user": user, "movie": movie, "rating": rating}
    # 原版user 等是int类型的


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


# 融合用户相似度：
def userbased_cal_matrix(matrix, test_user, test_movie):
    # STEP 1 计算test_user的均值
    test_user_vector = matrix[test_user]
    # test_user_avg = cal_avg(test_user_vector)  # test_user的评分均值
    # print test_user_vector
    # print test_user_avg
    # 只选为test_movie电影打过分的user
    sim_list = []
    for user in matrix:
        if test_movie in matrix[user]:  # 用户肯定评价过
            user_vector = matrix[user]
            # user_vector 得到 user train用户平价过的电影及其评分
            avg_sum = 0.0
            avg_num = 0
            test_avg_sum = 0.0
            test_avg_num = 0
            for key in test_user_vector:  # test_user_vector=matrix[test_user]
                if key in user_vector:  # user_vector = matrix[user]
                    avg_sum += float(user_vector[key])
                    avg_num += 1
                    test_avg_sum += float(test_user_vector[key])
                    test_avg_num += 1
            if avg_sum == 0.0:
                user_avg = 0.0
            else:
                user_avg = avg_sum/avg_num
            if test_avg_sum == 0.0:
                test_user_avg = 0.0
            else:
                test_user_avg = test_avg_sum/test_avg_num
            print(user, '用户的user_avg是：', user_avg)
            print(test_user, '测试用户的test_user_avg是：', test_user_avg)
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

        pearson_rating = molecusar / denominator
    else:
        # 如果没有相似集（本质上是因为没人看过这电影），取自己打分的平均值
        pearson_rating = cal_avg(test_user_vector)
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
        # print('hhh_test_timestamp；', test_timestamp)
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
        # pre_rating = weightSlopeOne(matrix, test_user, test_movie)  # 加权Slope One
        pre_rating = userbased_cal_matrix(matrix, test_user, test_movie)  # 融合用户相似度 Slope One
        print('预测分数：', pre_rating)
        t = pre_rating - float(test_rating)
        print('pre_rating:', pre_rating)
        print('test_rating:', test_rating)
        mae_sum += abs(t)
        rmae_sum += t * t
        number += 1
    print("MAE: " + str(mae_sum * 1.0 / number))
    print("RMAE: " + str(math.sqrt(rmae_sum * 1.0 / number)))


test()
