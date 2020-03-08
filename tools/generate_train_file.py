import os

path="/content/drive/My Drive/Diploma/datasets/CACD2000/"
agegrp_1=0
agegrp_2=0
agegrp_3=0
agegrp_4=0
agegrp_5=0
train_age_0=[]
train_age_1=[]
train_age_2=[]
train_age_3=[]
train_age_4=[]
test_age_0=[]
test_age_1=[]
test_age_2=[]
test_age_3=[]
test_age_4=[]

for fname in os.listdir(path):
    age = int(fname.split("_")[0])
    if (age >= 11 and age <= 20):
        agegrp_1 += 1
        if (agegrp_1 % 10 == 0):
            test_age_0.append(fname)
        else:
            train_age_0.append(fname)
    if (age >= 21 and age <= 30):
        agegrp_2 += 1
        if (agegrp_2 % 10 == 0):
            test_age_1.append(fname)
        else:
            train_age_1.append(fname)
    if (age >= 31 and age <= 40):
        agegrp_3 += 1
        if (agegrp_3 % 10 == 0):
           test_age_2.append(fname)
        else:
           train_age_2.append(fname)
    if (age >= 41 and age <= 50):
        agegrp_4 += 1
        if (agegrp_4 % 10 == 0):
            test_age_3.append(fname)
        else:
            train_age_3.append(fname)
    if (age >= 51):
        agegrp_5 += 1
        if (agegrp_5 % 10 == 0):
            test_age_4.append(fname)
        else:
            train_age_4.append(fname)

print("11-20",agegrp_1,len(test_age_0),len(train_age_0))
print("21-30", agegrp_2,len(test_age_1),len(train_age_1))
print("31-40",agegrp_3,len(test_age_2),len(train_age_2))
print("41-50", agegrp_4,len(test_age_3),len(train_age_3))
print("50+",agegrp_5,len(test_age_4),len(train_age_4))

with open(path + '../train.txt',"a+") as f:
    for fname in train_age_0:
        f.write("%s %d\n"%(fname, 0))
    for fname in train_age_1:
        f.write("%s %d\n"%(fname, 1))
    for fname in train_age_2:
        f.write("%s %d\n"%(fname, 2))
    for fname in train_age_3:
        f.write("%s %d\n"%(fname, 3))
    for fname in train_age_4:
        f.write("%s %d\n"%(fname, 4))
# with open('./age_data/train_data/train_age_group_0.txt',"a+") as f:
# for fname in train_age_0:
# f.write("%s\n"%fname)
#
# with open('./age_data/train_data/train_age_group_1.txt', "a+") as f:
# for fname in train_age_1:
# f.write("%s\n" %fname)
#
# with open('./age_data/train_data/train_age_group_2.txt', "a+") as f:
# for fname in train_age_2:
# f.write("%s\n" %fname)
#
# with open('./age_data/train_data/train_age_group_3.txt', "a+") as f:
# for fname in train_age_3:
# f.write("%s\n" %fname )
#
# with open('./age_data/train_data/train_age_group_4.txt', "a+") as f:
# for fname in train_age_4:
# f.write("%s\n" %fname )
#
# with open('./age_data/test_data/test_age_group_0.txt', "a+") as f:
# for fname in test_age_0:
# f.write("%s\n" % fname)
#
# with open('./age_data/test_data/test_age_group_1.txt', "a+") as f:
# for fname in test_age_1:
# f.write("%s\n" % fname)
#
# with open('./age_data/test_data/test_age_group_2.txt', "a+") as f:
# for fname in test_age_2:
# f.write("%s\n" % fname)
#
# with open('./age_data/test_data/test_age_group_3.txt', "a+") as f:
# for fname in test_age_3:
# f.write("%s\n" % fname)
#
# with open('./age_data/test_data/test_age_group_4.txt', "a+") as f:
# for fname in test_age_4:
# f.write("%s\n" % fname)
#
