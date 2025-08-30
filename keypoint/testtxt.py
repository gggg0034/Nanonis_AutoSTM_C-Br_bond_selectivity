import matplotlib.pyplot as plt

# 8个数，从左到右为第一个点的x,y,第二个点的x,y,第三个点的x,y,第四个点的x,y
points = [622.411 ,689.883 ,602.854 ,700.151 ,628.767 ,684.016 ,618.01 ,689.395]

# 创建一个751*751像素的白色背景
fig, ax = plt.subplots(figsize=(7.51, 7.51), facecolor='white')

# 绘制四个点
ax.plot(points[0], points[1], 'ro')
ax.plot(points[2], points[3], 'ro')
ax.plot(points[4], points[5], 'ro')
ax.plot(points[6], points[7], 'ro')

# 显示图形
plt.show()
 