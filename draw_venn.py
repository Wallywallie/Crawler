import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from matplotlib import rcParams
labels = ('谷德', 'ArchDaily', '有方')
only_gud = 743
only_archdaily = 679
only_youfang = 686
gud_archdaily = 22
gud_youfang = 15
archdaily_youfang = 79
all_three = 20

# 设置字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimSun']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题


# 创建 Venn 图
venn = venn3(subsets=(only_gud, only_archdaily, gud_archdaily, only_youfang, gud_youfang, archdaily_youfang, all_three),
             set_labels=labels)

# 设置每个区域的颜色为黑、白和灰
venn.get_patch_by_id('100').set_color('black')      # 只有谷德
venn.get_patch_by_id('010').set_color('gray')       # 只有 ArchDaily
venn.get_patch_by_id('001').set_color('lightgray')  # 只有有方
venn.get_patch_by_id('110').set_color('dimgray')    # 谷德和 ArchDaily
venn.get_patch_by_id('101').set_color('darkgray')   # 谷德和有方
venn.get_patch_by_id('011').set_color('silver')     # ArchDaily 和有方
venn.get_patch_by_id('111').set_color('gainsboro')  # 三者交集

# 去掉边框线条
for area in ['100', '010', '001', '110', '101', '011', '111']:
    patch = venn.get_patch_by_id(area)
    if patch:
        patch.set_edgecolor('none')

# 设置标题
plt.title("谷德、Archdaily和有方的数据重复关系")

# 显示图表
plt.show()