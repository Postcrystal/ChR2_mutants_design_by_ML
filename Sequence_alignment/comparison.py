import os

with open('../6eid_DE_records/trajectory0_seqs_1.txt', 'r') as f:
	seqs = f.readlines()	
	nums = 2

seqs_list = []
for i in range(nums):
	seq = str(seqs[i]).replace("\n", "")
	seqs_list.append(seq)

position = 0
for i in range(len(seqs_list[0])):
		#print(seqs_list[j][i])
		position += 1
		if seqs_list[0][i] != seqs_list[1][i]:
			print(position, seqs_list[0][i], seqs_list[1][i])
				
		

