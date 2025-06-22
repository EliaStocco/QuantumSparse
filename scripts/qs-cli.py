from quantumsparse.cli.scheduler import main,args_parser
parser = args_parser()
args = parser.parse_args()
main(args)