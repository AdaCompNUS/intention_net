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
if ! [ -d ${2} ]; then
    mkdir ${2}
fi
for d in "${1}/*/data/"; do
    cat ${d}label.txt >> ${2}"label.txt"
    mkdir ${2}rgb_0
    cp -r ${d}"rgb_0" ${2}
done
