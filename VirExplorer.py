
import os
import time
import csv
import numpy as np
from Bio import SeqIO
from tensorflow.keras.models import load_model
from more_itertools import chunked

os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(os.getenv('TF_GPU_ALLOCATOR'))

def encodeSeq5(seq) : 
    seq_code = list()
    for pos in range(len(seq)) :
        letter = seq[pos]
        if letter in ['A', 'a'] :
            code = [1,0,0,0,0]
        elif letter in ['C', 'c'] :
            code = [0,1,0,0,0]
        elif letter in ['G', 'g'] :
            code = [0,0,1,0,0]
        elif letter in ['T', 't'] :
            code = [0,0,0,1,0]
        else :
            code = [0,0,0,0,1]
        seq_code.append(code)
    return seq_code 

def batch_iterator(iterator, batch_size):
    """Returns lists of length batch_size.

    This can be used on any iterator, for example to batch up
    SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.AlignIO.parse(...), or simply
    lines from a file handle.

    This is a generator function, and it returns lists of the
    entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.
    """
    entry = True  # Make sure we loop once
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = next(iterator)  # iterator.next()
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch


start_time = time.time()

fileName = './test/xyz.fa'
output_dir = './output'
modDir = './models'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cutoff_len = 80
contigLength = 150
contigLengthk = str(contigLength / 1000)

#### Step 1: load model ####
print("1. Loading Models.")
print("   model directory {}".format(modDir))

## loading model and creating null dictionary for p-value evaluation
modDict = {}
nullDict = {}
modPattern = 'model_CNN_5Layers_' + contigLengthk + 'k'
modName = [x for x in os.listdir(modDir) if modPattern in x and x.endswith(".h5")][0]

load_model(os.path.join(modDir, modName))
modDict[contigLengthk] = load_model(os.path.join(modDir, modName))

Y_pred_file = [x for x in os.listdir(modDir) if modPattern in x and "Y_pred" in x][0]

with open(os.path.join(modDir, Y_pred_file)) as f:
    tmp = [line.split() for line in f][0]
    Y_pred = [float(x) for x in tmp]
Y_true_file = [x for x in os.listdir(modDir) if modPattern in x and "Y_true" in x][0]

with open(os.path.join(modDir, Y_true_file)) as f:
    tmp = [line.split()[0] for line in f]
    Y_true = [float(x) for x in tmp]
nullDict[contigLengthk] = Y_pred[:Y_true.index(1)]

model = modDict[contigLengthk]
null = nullDict[contigLengthk]

end_time1 = time.time() - start_time
print('Execution time prepaing files', end_time1 / 60)

print("2. Encoding Sequences.")

### check samples num output
cmd2 = 'grep ">" ' + fileName + ' | wc -l'
seqNumSmpld = os.system(cmd2)

numSeqsxFile = 10000  # number of sequence for each temporary file
NCBIName = os.path.splitext((os.path.basename(fileName)))[0]
fileDir = os.path.dirname(fileName)

contigLength = 150
contigLengthk = contigLength / 1000
if contigLengthk.is_integer():
    contigLengthk = int(contigLengthk)

outDir0 = fileDir
outDir = os.path.join(outDir0, "encode")
outDir2 = os.path.join(outDir, "temp")

if not os.path.exists(outDir):
    os.makedirs(outDir)
if not os.path.exists(outDir2):
    os.makedirs(outDir2)

# splitting input file in smaller files containing numSeqsxFile sequences
record_iter = SeqIO.parse(open(fileName), "fasta")
for i, batch in enumerate(batch_iterator(record_iter, numSeqsxFile)):
    filename = "split_%i_.fa_t" % (100 + i)
    with open(os.path.join(outDir2, filename), "w") as handle:
        count = SeqIO.write(batch, handle, "fasta")
    print("Wrote %i records to %s" % (count, filename))

fileList = [f for f in os.listdir(outDir2) if f.endswith('.fa_t')]
fileList.sort()
num_tot_seq = 0
j = 0
fileCount = 1
list_lftovr1 = []
list_lftovr2 = []

for file in fileList:
    print('Processing file: ', file)
    NCBIName = os.path.splitext((os.path.basename(file)))[0]
    codeFileNamefw = NCBIName+"#"+str(contigLengthk)+"k_"+"seq_"+"num-"+str(100+fileCount)+"_codefw.npy"    
     
    with open(os.path.join(outDir, codeFileNamefw), "bw") as f:
        
        records = SeqIO.parse(os.path.join(outDir2,file),"fasta")
        for batch in chunked(records, numSeqsxFile):
            list_seq_coded = []
            for seqRecord in batch:
                # print(seqRecord.id)
                # print(repr(seqRecord.seq))
                # print(len(seqRecord))
                if len(seqRecord) < contigLength:
                    print('Too short sequence!')
                    print(seqRecord.id)
                    print(repr(seqRecord.seq))
                    print(len(seqRecord))
                    list_lftovr1.append(seqRecord.id)
                    list_lftovr2.append(seqRecord.seq)
                    j += 1
                    continue
                countN = str(seqRecord.seq).count("N")
                if countN/len(seqRecord.seq) >= 0.3:
                    print('More than 30% N in sequence!')
                    print(seqRecord.id)
                    print(repr(seqRecord.seq))
                    print(len(seqRecord))
                    list_lftovr1.append(seqRecord.id)
                    list_lftovr2.append(seqRecord.seq)
                    j += 1
                    continue
                
                seq_coded = encodeSeq5(seqRecord.seq)
                list_seq_coded.append(seq_coded)
            
            np.save( f, np.array(list_seq_coded ,dtype = np.ubyte))
            num_tot_seq = num_tot_seq+ len(list_seq_coded)
        fileCount +=1     
        
    print("Encoded sequences are saved in {}".format(codeFileNamefw))
    print('Total number sequences deleted: {}'.format(j))   
    print('Total number sequences encoded: {}'.format(num_tot_seq))


end_time2 = time.time() - end_time1
print('Execution time encoding', end_time2 / 60)

print("3. Predicting Sequences.")
filenames_code = [x for x in os.listdir(outDir) if 'codefw.npy' in x and str(contigLengthk) + 'k' in x]
filenames_code.sort()
filenames_seq = [x for x in os.listdir(outDir) if 'seq.fasta' in x and str(contigLengthk) + 'k' in x]
filenames_seq.sort()
# clean the output file
outfile = os.path.join(output_dir, os.path.basename(fileName) + '_gt' + str(cutoff_len) + 'bp_dvfpred.txt')
predF = open(outfile, 'w')
writef = predF.write('\t'.join(['name', 'len', 'score', 'pvalue']) + '\n')
predF.close()
predF = open(outfile, 'a')

### generating outpu file with all predicitions scores
for fname_code, fname_seq in zip(filenames_code, filenames_seq):
    temp = np.array([np.load(os.path.join(outDir, fname_code))], dtype=np.float16)[0]
    score = model.predict(temp)
    score1 = score[:, 0]
    pvalue = sum([x > score for x in null]) / len(null)
    pvalue1 = pvalue[:, 0]
    head = []
    seqL = []
    for seqRecord in SeqIO.parse(os.path.join(outDir, fname_seq), "fasta"):
        head.append(seqRecord.id)
        seqL.append(len(seqRecord))

    with open(outfile, "at") as file:
        writer = csv.writer(file, delimiter='\t')
        for row in zip(head, seqL, score1, pvalue1):
            writer.writerow(row)
    endTime = time.time() - start_time
    print('Execution time writing file', endTime / 60)

predF.close()

end_time3 = time.time() - end_time2
print('Execution time predicitng', end_time3 / 60)
end_time_tot = time.time() - start_time
print('Total execution time', end_time_tot / 60)


