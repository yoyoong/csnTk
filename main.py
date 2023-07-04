import argparse, sys
import csn

version = '1.0'

class HelpMessage():
    csn = "Calculate the CSN statistic and generate the NDM matrix of sample's gene expression matrix"
    inputFile = "inputdata gene expression file, .gz format"
    gene1 = "the first inputdata gene name list, txt format, one gene name per line"
    gene2 = "the second inputdata gene name list, txt format, one gene name per line"
    sampleID = "inputdata sample ID list, txt format, one sample ID per line"
    boxSize = "Size of neighborhood, Default = 0.1 (nx(k) = ny(k) = 0.1*n)"
    alpha = "Significant level (eg. 0.001, 0.01, 0.05 ...), Default = 0.01"
    tag = "prefix of the output file(s)"
    ndmFlag = "whether generate NDM matrix"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help = HelpMessage()

    subparsers1 = parser.add_subparsers()
    gw = subparsers1.add_parser('csn', help=help.csn)
    gw.add_argument('--inputFile', type=str, required=True, help=help.inputFile)
    gw.add_argument('--gene1', type=str, required=False, help=help.gene1)
    gw.add_argument('--gene2', type=str, required=False, help=help.gene2)
    gw.add_argument('--sampleID', type=str, required=False, help=help.sampleID)
    gw.add_argument('--boxSize', type=float, required=False, help=help.boxSize, default=0.1)
    gw.add_argument('--alpha', type=float, required=False, help=help.alpha, default=0.01)
    gw.add_argument('--tag', type=str, required=True, help=help.tag)
    gw.add_argument('--ndmFlag', action='store_true', required=False, help=help.ndmFlag)
    gw.set_defaults(func='csn')

    args = parser.parse_args()
    try:
        args.func
    except:
        print('csnTk: A comprehensive tool kit for analysis of CSN')
        print('version:', version)
        sys.exit()

    if args.func == 'csn':
        csn.main(args)
    else:
        print("Unrecognized command: " + args.func)
        sys.exit()