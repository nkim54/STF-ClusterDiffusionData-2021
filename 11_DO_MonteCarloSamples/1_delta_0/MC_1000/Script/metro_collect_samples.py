import json


folder = '1500K'

sets = 10
file_folder = ''
file_sample = 'metro_sample'



metro_sample = []
for i in range(sets):
    file_name = folder+'/'+file_folder+str('{:02d}'.format(i))+'/'+file_sample+'.txt'
    with open(file_name, 'r') as f:
        data = json.load(f)

    metro_sample += data

metro_sample.sort()



file_out = file_sample + '_' + folder + '.txt'
dat_out = metro_sample

with open(file_out, 'w') as f:
    json.dump(dat_out, f, indent=2, sort_keys=True)