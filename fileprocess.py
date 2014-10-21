#!/usr/bin/python

import sys

def readFile(fname):
	i = 0
	fname1 = fname + "_1"
	fname2 = fname + "_2"
	fname3 = fname + "_3"
	fname4 = fname + "_4"

	print fname1
	f1 = open(fname1,'w')
	f2 = open(fname2,'w')
	f3 = open(fname3,'w')
	f4 = open(fname4,'w')
	
	headerLine = ""
	with open(fname) as f:
    		content = f.readlines()
		for line in content:
			line.strip()
			if(i==0):
				i = i + 1
				headerLine = line
				f1.write(line+'\n')
				f2.write(line+'\n')
				f3.write(line+'\n')
				f4.write(line+'\n')
				continue
			i = i + 1
	
			fields = line.split('\t')
			print fields[3], fields[15]
			if((fields[3].strip() == 'AOS') and (fields[15].strip() == '4')):
				f1.write(line+'\n') 
			if((fields[3].strip() == 'AOS') and (fields[15].strip() == '8')):
				f2.write(line+'\n') 
			if((fields[3].strip() == 'SOA') and (fields[15].strip() == '4')):
				f3.write(line+'\n') 
			if((fields[3].strip() == 'SOA') and (fields[15].strip() == '8')):
				f4.write(line+'\n') 
	f1.close()
	f2.close()
	f3.close()
	f4.close()
			
			

if __name__ == "__main__":
        dataFile = sys.argv[1]
	readFile(dataFile)
	
