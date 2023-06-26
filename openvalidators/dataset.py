# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import random
import bittensor as bt
from datasets import load_dataset
from collections.abc import Iterator

def chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


import os
import json
import torch
import random
import requests
import bittensor
import queue
from tqdm import tqdm
from os import listdir
from os.path import isfile, join, isdir
from typing import List
from collections.abc import Iterator


def chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

CHUNK_SIZE: int = 1000

class Mountain():
    def __init__( self, download_folder: str = "~/tmp/dataset/text/mountain/" ):
        self.download_folder = os.path.expanduser( download_folder )
        self.__folder_links = None
        self.__filenames = None
        self.__index = None
        self.__length = None
        self.download()
        self.__index = self.index

    def get_text( self, idx: int ) -> str:
        response = requests.post("http://global.ipfs.opentensor.ai/api/v0/cat?arg={}".format( self.__getitem__( idx ) ))
        return response.content.decode('utf-8')

    def __len__( self ):
        return self.length

    def __getitem__( self, idx: int ):
        counter = 0
        for folder_dir, folder_stats in self.index.items():
            if counter + folder_stats['total_hashes'] > idx:
                file_idx = int( (idx - counter) / CHUNK_SIZE )
                file_pos = int( (idx - counter) % CHUNK_SIZE )
                return self._hashes_for_file( folder_dir + "/chunk_{}.txt".format( file_idx ) )[file_pos]['Hash']
            counter += folder_stats['total_hashes']
        raise ValueError("Index out of bounds (0, {})".format( counter ))
            
    @property
    def length(self):
        if self.__length == None:
            self.__length = sum( [ val['total_hashes'] for val in self.__index.values() ] )
        return self.__length

    def download ( self, force_download: bool = False ):
        if self.is_downloaded and not force_download: return 
        for folder_link in tqdm( self.folder_links ): 
            if not self._is_folder_link_downloaded( folder_link ) and not force_download:
                self._download_folder_link( folder_link )

    @property
    def index( self ):
        if self.__index == None:
            self.__index = {}
            if not self.is_downloaded: 
                self.download()
            for folder_link in self.folder_links:
                folder_files: List[str] = self._files_for_folder_link( folder_link )
                if len(folder_files) == 0: continue 
                n_files = len( folder_files )
                preffix_size = len( self._hashes_for_file( folder_files[ 0 ] ) )
                suffix_size = len( self._hashes_for_file( folder_files[ -1 ] ) )
                total_hashes = (n_files - 1) * preffix_size + suffix_size
                self.__index[ self._dir_for_folder_link( folder_link )  ] = {'n_files': n_files, 'total_hashes': total_hashes, 'suffix_size': suffix_size }
        return self.__index

    @property
    def is_downloaded( self ) -> bool:
        return all( [ self._is_folder_link_downloaded( folder_link ) for folder_link in self.folder_links ] )

    @property
    def folder_links( self ) -> List[ dict ] :
        if self.__folder_links == None:
            response = requests.post("http://global.ipfs.opentensor.ai/api/v0/object/get?arg=QmSdDg6V9dgpdAFtActs75Qfc36qJtm9y8a7yrQ1rHm7ZX")
            self.__folder_links = response.json()['Links']
        return self.__folder_links

    @property
    def folder_dirs( self ) -> List[ dict ] :
        return [ join(self.download_folder, f) for f in listdir(self.download_folder) if isdir(join(self.download_folder, f)) ]

    @property
    def filenames( self ) -> List[ str ] :
        if self.__filenames == None:
            self.__filenames  = []
            for folder_link in self.folder_links: 
                self.__filenames.extend( self._files_for_folder_link( folder_link )) 
        return self.__filenames


    def _files_for_folder_link(self, folder_link: dict ) -> List[ str ]: 
        if not self._is_folder_link_downloaded( folder_link ): self._download_folder_link( folder_link )
        folder_dir = self._dir_for_folder_link( folder_link )
        return [ join(folder_dir, filename) for filename in listdir(folder_dir) if isfile(join(folder_dir, filename)) ]

    def _hashes_for_file(self, filename: str ) -> List[ dict ]: 
        with open(filename, 'r') as f:
            return json.loads( f.read() )

    def _download_folder_link( self, folder_link: dict ):
        dirname = self._dir_for_folder_link( folder_link )
        os.makedirs( dirname, exist_ok=True )
        response = requests.post("http://global.ipfs.opentensor.ai/api/v0/cat?arg={}".format( folder_link['Hash'] ))
        hashes_dict = json.loads(response.content)
        for i, chunk_i in enumerate( chunk( hashes_dict, CHUNK_SIZE ) ):
            filename = dirname + "/chunk_{}.txt".format( i )
            with open(filename, 'w') as f: f.write(json.dumps( chunk_i ))

    def _is_folder_link_downloaded( self, folder_link:dict ) -> bool:
        return os.path.exists( self._dir_for_folder_link( folder_link ) )

    def _dir_for_folder_link( self, folder_link: dict ) -> str:
        return os.path.expanduser( self.download_folder ) + "/" + Mountain._name_for_folder_link( folder_link ) 

    @staticmethod
    def _name_for_folder_link( folder_link: dict ) -> str:
        return folder_link['Name'].split(".txt")[0]

class Dataset(Iterator):
    def __init__( self, num_words: int = 300 ):
        self.num_words = num_words
        self.mountain_hashes = Mountain()

    def __next__( self ):
        tokens = []
        index = random.randint( 0, len( self.mountain_hashes ) )
        filehash = self.mountain_hashes[ index ]
        response = requests.post("http://global.ipfs.opentensor.ai/api/v0/cat?arg={}".format( filehash ))
        text =response.content.decode('utf-8') 
        next_split = text.split(" ")
        chunks = list(chunk(next_split, self.num_words))
        joined_words = random.choice(chunks).join(" ")
        return { "text": joined_words }

# class Dataset(Iterator):
#     def __init__(self):
#         super().__init__()
#         seed = random.randint(0,1000)
#         self.openwebtext = iter( load_dataset("openwebtext", split="train", streaming=True).shuffle(seed=seed, buffer_size=100000) )
#         self.red_pajama = iter( load_dataset("togethercomputer/RedPajama-Data-1T").shuffle(seed=seed, buffer_size=100000) )

#     def __next__(self):
#         if random.random() < 0.5:
#             return {"text": next(self.openwebtext)["text"]}
#         elif random.random() > 0.5:
#             # Red Pajama is a bit too long, so we split it into chunks of 300 words.
#             next_text = next(self.red_pajama)["text"]
#             next_split = next_text.split(" ")
#             chunks = list(chunk(next_split, 300))
#             return {"text": random.choice(chunks)}


class MockDataset(Iterator):
    def __next__(self):
        return {"text": "What is the capital of Texas?"}


   