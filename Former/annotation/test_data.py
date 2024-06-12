import os
import glob


def replace_path():
    all_txt = glob.glob('set_*.txt')
    print(all_txt)
    frame16_dir = '/home/s5u1/mafuyan/Dataset/DFEW/clip_224x224_16f/'
    for i in all_txt:
        tmp = [x.strip().split(' ') for x in open(i)]
        new_file = 'ma_'+i.split('.')[0]+'.txt'
        f = open(new_file,'a')
        for item in tmp:
            if len(glob.glob(os.path.join(item[0], '*.jpg')))<=16:
                f.write(item[0].replace("/home/s5u1/mafuyan/Dataset/DFEW/images/",frame16_dir) + ' ' + str(16) + ' ' + item[2] + "\n")
                # print(1)
            else:
                f.write(item[0]+' '+str(len(glob.glob(os.path.join(item[0], '*.jpg'))))+' '+item[2]+"\n")
        f.close()

# def test_path():
#     all_txt = glob.glob('set_*.txt')
#     print(all_txt)
#     for i in all_txt:
#         tmp = [x.strip().split(' ') for x in open(i)]
#         for item in tmp:
#             if len(glob.glob(os.path.join(item[0],'*.jpg'))) != int(item[1]):
#                 print(item[0])


# tmp = [x.strip().split(' ') for x in open('set_1_train.txt')]
# # tmp = [item for item in tmp if int(item[1]) >= 16]
# for item in tmp:
#     if len(glob.glob(os.path.join(item[0],'*.jpg'))) != int(item[1]):
#         print(item[0])

emo_dict={
'Surprise':'4',
'Neutral':'2',
'Sad':'1',
'Angry':'3',
'Disgust':'5',
'Happy':'0',
'Fear':'6'
}

def write_afew():
    data_path = "/home/s5u1/mafuyan/Dataset/AFEW_processed/train"
    f = open("ma_train_afew.txt", 'a')
    for cate in os.listdir(data_path):
        # f.writelines()
        for file in os.listdir(os.path.join(data_path,cate)):
            dir = os.path.join(data_path,cate,file)
            f.write(dir + ' ' + str(len(glob.glob(os.path.join(dir, '*.jpg')))) + ' ' + emo_dict[cate] + "\n")
    f.close()


if __name__ == '__main__':
    replace_path()