# To run the bag file the folder architecture must be
# data_1 data_2 data_3 ,.... where
# data_i
# |----data
#      |----label.txt
#      |----rgb_0
#      |----rgb_1
#      |----rgb_2
# |----a.bag
# |----b.bag
for d in "${1}/*/"; do
    sed -e 1d ${d}label2.txt >> test
done