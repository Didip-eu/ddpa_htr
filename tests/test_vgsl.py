import pytest
import sys
import torch
from pathlib import Path

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

from kraken import vgsl, layers


# basic VGSL specs

def test_strict_spec_input():
    net = vgsl.TorchVGSLModel('[1,2,3,4]').nn
    assert isinstance(net, layers.MultiParamSequential)


def test_strict_spec():
    net = vgsl.TorchVGSLModel('[4,256,4096,3 ([Cr3,3,32 Do0.1,2 Mp2,2] [Cl3,3,64 Mp2,2]) S1(1x0)1,3 Lbx200 O1c42]').nn
    assert isinstance(net, layers.MultiParamSequential)


def test_spec_with_spaces():
    net = vgsl.TorchVGSLModel(' [ 4,256,4096,3 ( [  Cr3,3,32 Do0.1,2  Mp2,2]   [ Cl3,3,64 Mp2,2] )  S1( 1x0)1,3 Lbx200 O1c42  ] ').nn
    assert isinstance(net, layers.MultiParamSequential)


def test_strict_spec():
    net = vgsl.TorchVGSLModel('[4,256,4096,3 ([Cr3,3,32 Do0.1,2 Mp2,2] [Cl3,3,64 Mp2,2]) S1(1x0)1,3 Lbx200 O1c42]').nn
    assert isinstance(net, layers.MultiParamSequential)


def test_resnet_block_syntax():
    net = vgsl.TorchVGSLModel('[ 4,256,4096,3 Rn32 ]').nn
    assert isinstance(list(net.children())[0], layers.ResNetBasicBlock)

def test_resnet_block_output_tensor():

    net = vgsl.TorchVGSLModel('[ 4,256,4096,3 Rn32 ]').nn
    assert net( torch.rand((4,3,256,4096)))[0].shape == (4,32,256,4096)

def test_resnet_block_output_length():

    net = vgsl.TorchVGSLModel('[ 4,256,4096,3 Rn32 ]').nn
    lengths = torch.tensor([3072,4096,2048,4096])
    assert torch.equal( net( torch.rand((4,3,256,4096)), lengths)[1], lengths)

def test_resnet_block_chain():
    net = vgsl.TorchVGSLModel('[4,128,0,3 Cr7,7,32 Mp2,2 Rn64 Rn64 Mp2,2 Rn128 Rn128 Rn128 Rn128 Mp2,2 Rn256 Rn256 Rn256 Rn256]').nn
    assert net( torch.rand((4,3,128,1024)))[0].shape == (4,256,16,128)

def test_addition_vgsl():
    """
    Testing that the (undocumented) Addition block does what it is supposed to do,
    i.e. adding two slices of a tensor.
    """
    chunk_size = 32
    # 2 parallel convolutions: (N,C,H,W) -> (N,2C,H,W) 
    parallelConvOp = vgsl.TorchVGSLModel(f'[0,0,0,3 (Cl3,3,32 Cl3,3,{chunk_size})]').nn
    # adding two slices (N,C,H,W) in C dimension (= dim '3' in VGSL)
    additionOp = vgsl.TorchVGSLModel(f'[0,0,0,64 A3,{chunk_size}]').nn
    t1 = parallelConvOp(torch.rand((4,3,128,1024)))[0]
    t2 = additionOp( t1 )[0]
    t3 = t1[:,:chunk_size] + t1[:,chunk_size:]
    assert torch.equal( t2, t3)


def test_build_spec_from_chunks():

    
    spec = vgsl.build_spec_from_chunks( [ ('Input','0,0,0,3'),
          ('CNN Backbone', 'Cr7,7,32 Mp2,2 Rn64 Rn64 Mp2,2 Rn128 Rn128 Rn128 Rn128 Mp2,2 Rn256 Rn256 Rn256 Rn256'),
          ('Column Maxpool', 'Mp{height//8},1'),
          ('Recurrent head', 'Lbx256') ], 
          height=128 )
    assert spec == '[0,0,0,3 Cr7,7,32 Mp2,2 Rn64 Rn64 Mp2,2 Rn128 Rn128 Rn128 Rn128 Mp2,2 Rn256 Rn256 Rn256 Rn256 Mp16,1 Lbx256]'

    

    # 



