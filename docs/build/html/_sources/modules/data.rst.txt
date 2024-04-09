multimodal_transformers.data
===============================

The data module includes two functions to help load your own datasets
into :class:`multimodal_transformers.data.tabular_torch_dataset.TorchTabularTextDataset`
which can be fed into a :class:`torch.utils.data.DataLoader`. The
:obj:`multimodal_transformers.data.tabular_torch_dataset.TorchTabularTextDataset`'s
:obj:`__getitem__` method's outputs can be directly fed to the
forward pass to a model in :obj:`multimodal_transformers.model.tabular_transformers`.

.. Note::
    You may still need to move the :obj:`__getitem__` method outputs to the right gpu device.


Module contents
----------------

.. automodule:: multimodal_transformers.data
   :members:
   :undoc-members:
   :show-inheritance:
