import camera as camera
import cv2
import os


def generate(dirname):
    #1.加载人脸分类器
    face_cascade = cv2.CascadeClassifier('C:/Users/13956/Desktop/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:/Users/13956/Desktop/opencv/sources/data/haarcascades/haarcascade_eye.xml')
    # 带眼镜的时候可以用下面这个
    # eye_cascade = cv2.CascadeClassifier('C:/Users/13956/Desktop/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

    # 创建目录
    if (not os.path.isdir(dirname)):#判断是否为目录
        os.makedirs(dirname)#如果不是目录则创建目录

    # 打开摄像头进行人脸图像采集
    camera = cv2.VideoCapture(0)#表示打开笔记本的内置摄像头
    count = 0
    while (True):
        ret, frame = camera.read()#读取一帧的图片,ret获取返回值，frame是图片
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#将BGR格式转换成灰度图片
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #利用训练好的数据识别，参数:输入图像  人脸目标序列  每次图像减小的比例  存储最小真识别认可   最小尺寸  最大尺寸
        #最后三个参数都可以降低误差
        for (x, y, w, h) in faces:#x,y是识别到的坐标，w,h是识别到的范围宽度，高度
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #rectang用来绘制矩形框 通常用在图片的标记上 参数：1.被处理图片 2.3矩形左上右下角坐标  4.颜色   5.线型
            # 重设置图像尺寸
            #200 * 200
            f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            cv2.imwrite(dirname + '/%s.pgm' % str(count), f)#写入文件
            print(count)
            count += 1#迭代器++

        cv2.imshow("camera", frame)
        if cv2.waitKey(100) & 0xff == ord("q"):
        #为何做掩码运算是因为已经发现在Linux中的某些情况下（when OpenCV uses GTK as its backend GUI），waitKey()可能返回超过ASCII的keycode，所以这是为了防止在某些情况下产生bug。
            break
    # 下面是你想要多少张图片就停止
        elif count > 20:
            break


    camera.release()#释放相机
    cv2.destroyAllWindows()#释放窗口

if __name__ == "__main__":
    #__name__是python的一个内置类属性，是标识模块的名字的一个系统变量。
    #如果当前模块被直接执行（主模块），__name__存储的是__main__

    #如果当前模块是被调用的模块（被导入），则__name__存储的是py文件名（模块名称）
    #在所有代码执行之前，__name__ 变量值被设置为 '__main__' 
    generate("C:/Users/13956/Desktop/666") # 你生成的图片放在的电脑中的地方，调用函数
