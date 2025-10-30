"""
Command Line Interface Module
Implements CLI for the neuroscience modeling framework
"""
from .interface import Interface
import argparse
import sys


class InterfaceCLI(Interface):
    """Command Line Interface implementation"""
    
    def __init__(self):
        super().__init__()
        self.registry = None
        self.model_registry = None
    
    def setup(self) -> None:
        """Initialize the CLI interface"""
        print("Setting up CLI Interface...")
        self.parser = argparse.ArgumentParser(
            description='Neuroscience Modeling Framework'
        )
        self.parser.add_argument(
            '--model',
            type=str,
            help='Model class to use (e.g., Model1, Model2)',
            required=False
        )
        self.parser.add_argument(
            '--mode',
            type=str,
            choices=['train', 'validate', 'test'],
            help='Operation mode',
            default='train'
        )
        self.parser.add_argument(
            '--config',
            type=str,
            help='Path to configuration file',
            required=False
        )
    
    def run(self, name: str) -> bool:
        """
        Run the CLI interface
        
        Args:
            name: Name of the run
            
        Returns:
            bool: Success status
        """
        print(f"Running CLI Interface: {name}")
        
        try:
            args = self.parser.parse_args()
            print(f"Mode: {args.mode}")
            print(f"Model: {args.model if args.model else 'default'}")
            print(f"Config: {args.config if args.config else 'none'}")
            
            return args
        except SystemExit:
            return None
    
    def display_help(self):
        """Display help information"""
        self.parser.print_help()
