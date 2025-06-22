from typing import Union, Dict, List, Any
from dataclasses import dataclass, field

#---------------------------------------#
@dataclass
class Description:
    
    spin: List[int]
    
#---------------------------------------#
@dataclass
class Scheduler:
    
    name:str
    description: Description
    tasks:  List[str]
    
    @staticmethod
    def from_yaml(file: Union[Dict[str,Any],str]) -> "Scheduler":
        if isinstance(file,str):
        
            import os
            if not os.path.exists(file):
                raise ValueError(f"File '{file}' does not exist.")
            
            import yaml
            try:
                with open(file, "r") as f:
                    data = yaml.safe_load(f)
            except Exception as err:
                raise ValueError(f"Error while reading file '{file}': {err}")
        elif isinstance(file,dict):
            data = file
        else:
            raise TypeError(f"Only str and dict supported, while provided {type(file)}.")
        return Scheduler(**data)
    
#---------------------------------------#
@dataclass
class Tasks:
    
    tasks: List[str]
        
#---------------------------------------#
def args_parser(description="CLI by qunatumsparse"):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"        , **argv, required=True , type=str, help="input file [yaml]")
    return parser
        
#---------------------------------------#
def main(args=None):
    
    if args is None:
        args = args_parser().parse_args()
    
    sch = Scheduler.from_yaml(args.input)
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()