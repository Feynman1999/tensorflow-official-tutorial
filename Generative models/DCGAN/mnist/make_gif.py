import matplotlib.pyplot as plt
import imageio, os


def create_gif(image_list, gif_name, duration = 0.3):
    """
    生成gif文件，原始图像仅仅支持png格式
    gif_name : 字符串，所生成的gif文件名，串过来时已经带.gif文件名后缀；
    duration : gif图像时间间隔，这里默认设置为1s,当然你喜欢可以设置其他；
    """
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # 保存为gif格式的图
    imageio.mimsave(gif_name, frames, 'GIF', duration = duration)
    return


if __name__ == '__main__':
    TIME_GAP=0.075
    filepath = './images'
    name_list = []
    for filename in os.listdir(filepath):
        name_list.append(os.path.join(filepath,filename))

    name_list.sort()
    # for name in name_list:
    #     print(name) 
    create_gif(name_list, 'demo.gif' , TIME_GAP)




