from scipy.sparse import csr_matrix
from scipy.stats import norm
import numpy as np
import pandas as pd
import sys, os

def upperlower(data, boxsize):
    (n1, n2) = data.shape  # n1 gene; n2 sample
    upper = np.zeros((n1, n2), dtype=np.float32)
    lower = np.zeros((n1, n2), dtype=np.float32)

    for i in range(0, n1):
        s1 = sorted(data[i, :])
        s2 = data[i, :].argsort()
        h = round(boxsize / 2 * n2)
        k = 0
        while k < n2:
            s = 0
            while k + s + 1 < n2:
                if s1[k + s + 1] == s1[k]:
                    s = s + 1
                else:
                    break

            if s >= h:
                upper[i, s2[k:k + s + 1]] = data[i, s2[k]]
                lower[i, s2[k:k + s + 1]] = data[i, s2[k]]
            else:
                upper[i, s2[k:k + s + 1]] = data[i, s2[min(n2 - 1, k + s + h)]]
                lower[i, s2[k:k + s + 1]] = data[i, s2[max(0, k - h)]]

            k = k + s + 1
    return (upper, lower)


# def getCSNList(args, input_data):
#     data, gene, sample = input_data.values, input_data.index.values, input_data.columns.values
#     (n1, n2) = data.shape
#     eps = np.finfo(float).eps
#
#     (upper, lower) = upperlower(data, boxsize=args.boxSize)
#
#     csn_writer = open(args.tag + ".CSN.txt", "w")
#     csn_writer.writelines("gene1\tgene2\tsampleID\tvalue\n")
#
#     ndm = pd.DataFrame(np.zeros((len(gene), n2)), index=gene, columns=sample)
#
#     for k in range(0, n2):
#         sampleID = sample[k]
#         B = np.zeros((n1, n2), dtype=np.float32)
#         for j in range(0, n2):
#             B[:, j] = (data[:, j] <= upper[:, k]) & (data[:, j] >= lower[:, k]) & (data[:, k] > 0)
#         a = B.sum(axis=1)
#         a = np.reshape(a, (n1, 1))
#         # temp = (np.dot(B, B.T) * n2 - np.dot(a, a.T)) / np.sqrt(np.dot(a, a.T) * np.dot(n2 - a, (n2 - a).T) / (n2 - 1) + eps)
#         temp = (B @ B.T * n2 - a @ a.T) / np.sqrt((a @ a.T) * ((n2 - a) @ (n2 - a).T) / (n2 - 1) + eps)
#
#         np.fill_diagonal(temp, 0)
#         matrix = csr_matrix(temp).tocoo()
#         for index in zip(matrix.nonzero()[0], matrix.nonzero()[1]):
#             if (index[0] < index[1]) and (temp[index[0]][index[1]] > norm.ppf(1 - args.alpha)):
#                 gene1_name = gene[index[0]]
#                 gene2_name = gene[index[1]]
#                 ndm.iloc[index[0], k] += 1
#                 ndm.iloc[index[1], k] += 1
#                 value = temp[index[0]][index[1]]
#                 csn_writer.writelines(str(gene1_name) + "\t" + str(gene2_name) + "\t" + str(sampleID) + "\t" + str(value) + "\n")
#         print(str(sampleID) + "end!")
#     if args.ndmFlag:
#         ndm.astype(int).to_csv(args.tag + ".NDM.txt", sep='\t')
#     csn_writer.close()

def getCSNList(args, input_data1, input_data2):
    data1, gene1, sample1 = input_data1.values, input_data1.index.values, input_data1.columns.values
    data2, gene2, sample2 = input_data2.values, input_data2.index.values, input_data2.columns.values
    gene = list(np.union1d(gene1, gene2)) # merge gene1 and gene2
    n1, n11, n12, n2 = len(gene), data1.shape[0], data2.shape[0], sample1.shape[0]
    eps = np.finfo(float).eps

    (upper1, lower1) = upperlower(data1, boxsize=args.boxSize)
    (upper2, lower2) = upperlower(data2, boxsize=args.boxSize)

    csn_writer = open(args.tag + ".CSN.txt", "w")
    csn_writer.writelines("gene1\tgene2\tsampleID\tvalue\n")

    ndm = pd.DataFrame(np.zeros((n1, n2)), index=gene, columns=sample1)

    for k in range(0, n2):
        sampleID = sample1[k]
        B1 = np.zeros((n11, n2), dtype=np.float32)
        B2 = np.zeros((n12, n2), dtype=np.float32)
        for j in range(0, n2):
            B1[:, j] = (data1[:, j] <= upper1[:, k]) & (data1[:, j] >= lower1[:, k]) & (data1[:, k] > 0)
            B2[:, j] = (data2[:, j] <= upper2[:, k]) & (data2[:, j] >= lower2[:, k]) & (data2[:, k] > 0)
        a1 = np.reshape(B1.sum(axis=1), (n11, 1))
        a2 = np.reshape(B2.sum(axis=1), (n12, 1))
        temp = (B1 @ B2.T * n2 - a1 @ a2.T) / np.sqrt((a1 @ a2.T) * ((n2 - a1) @ (n2 - a2).T) / (n2 - 1) + eps)

        np.fill_diagonal(temp, 0)
        matrix = csr_matrix(temp).tocoo()
        written_overlap_gene = set() # overlap gene(exists in both gene1 and gene2) which has been written
        for index in zip(matrix.nonzero()[0], matrix.nonzero()[1]):
            gene1_name = gene1[index[0]]
            gene2_name = gene2[index[1]]
            value = temp[index[0]][index[1]]
            if (gene1_name != gene2_name) and (value > norm.ppf(1 - args.alpha)):
                if gene2_name in gene1:
                    if gene2_name not in written_overlap_gene: # overlap gene only write one side
                        ndm.iloc[gene.index(gene1_name), k] += 1
                        ndm.iloc[gene.index(gene2_name), k] += 1
                        csn_writer.writelines(str(gene1_name) + "\t" + str(gene2_name) + "\t" +
                                              str(sampleID) + "\t" + str(value) + "\n")
                        written_overlap_gene.add(gene1_name)
                else:
                    ndm.iloc[gene.index(gene1_name), k] += 1
                    ndm.iloc[gene.index(gene2_name), k] += 1
                    csn_writer.writelines(str(gene1_name) + "\t" + str(gene2_name) + "\t" +
                                          str(sampleID) + "\t" + str(value) + "\n")
        print("Sample: " + str(sampleID) + " get csn end!")
    if args.ndmFlag:
        ndm.astype(int).to_csv(args.tag + ".NDM.txt", sep='\t')
    csn_writer.close()

def main(in_args):
    args = in_args
    try:
        basename = os.path.basename(args.inputFile)
        if basename.find('.pickle') >= 0:
            input_data = pd.read_pickle(args.inputFile)
        else:
            input_data = pd.read_csv(args.inputFile, sep='\t')
    except:
        sys.stderr.write('Fail to open input file %s' % args.inputFile)
        sys.exit(1)
    print(input_data)
    if args.sampleID is not None:
        input_sampleID = []
        with open(args.sampleID, 'r') as f:
            for item in f.readlines():
                input_sampleID.append(item.strip())
        input_data = input_data.loc[:, input_sampleID]

    if args.gene1 is not None and args.gene2 is not None:
        input_gene1 = []
        with open(args.gene1, 'r') as f:
            for item in f.readlines():
                input_gene1.append(item.strip())
        input_data1 = input_data.loc[input_gene1, :]

        input_gene2 = []
        with open(args.gene2, 'r') as f:
            for item in f.readlines():
                input_gene2.append(item.strip())
        input_data2 = input_data.loc[input_gene2, :]

        getCSNList(args, input_data1, input_data2)
    else:
        getCSNList(args, input_data, input_data)
