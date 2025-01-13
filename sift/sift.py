
def img_similarity():
    import cv2 as cv
    import os
    curr_folder = os.getcwd()
    sub_folder = os.path.join(curr_folder, 'sift')
    path_1 = os.path.join(sub_folder, 'img_gd','021.jpg')
    path_2 = os.path.join(sub_folder, 'img_gd','022.jpg')


    img_1 = cv.imread(path_1)
    img_2 = cv.imread(path_2)

    sift = cv.SIFT.create()

    # 特征点提取与描述子生成
    kp1, des1 = sift.detectAndCompute(img_1,None)
    kp2, des2 = sift.detectAndCompute(img_2,None)

    # 暴力匹配
    bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
    matches = bf.knnMatch(des1,des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    good_match_ratio = len(good_matches) / len(matches)
    print(len(good_matches))
    print(len(matches))

    # 打印相似度结果

    print(f"Good Match Ratio: {good_match_ratio * 100:.2f}%")

    img_matches = cv.drawMatches(img_1, kp1, img_2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 显示匹配的结果
    cv.imshow("Good Matches", img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()


img_similarity()
