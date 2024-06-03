                        Training Loop
+--------------------------------------------------------------+
|                          Epoch Loop                          |
|                                                              |
| +----------------------------------------------------------+ |
| |                         Batch Loop                        | |
| |                                                          | |
| | +--------------------------------------+                 | |
| | |           Get Batch                  |                 | |
| | |                                      |                 | |
| | | - Input: source (Tensor)             |                 | |
| | | - Process:                           |                 | |
| | |   seq_len = min(bptt, len(source) - 1 - i)              | |
| | |   data: shape [seq_len, batch_size]                    | |
| | |   target: shape [seq_len * batch_size]                 | |
| | | - Output: (data, target)                               | |
| | +------------------------|-------------+                 | |
| |                          |                               | |
| |                          v                               | |
| | +--------------------------------------+                 | |
| | |          Forward Pass                |                 | |
| | |                                      |                 | |
| | | - Input: data (Tensor)               |                 | |
| | | - Output: output (Tensor)            |                 | |
| | | - output shape: [seq_len, batch_size, ntokens]         | |
| | +------------------------|-------------+                 | |
| |                          |                               | |
| |                          v                               | |
| | +----------------------------------------+               | |
| | |         Compute Loss                   |               | |
| | |                                        |               | |
| | | - Input: output, targets               |               | |
| | | - Output: loss (Tensor)                |               | |
| | | - targets: shape [seq_len * batch_size]|               | |
| | | - output_flat: [seq_len * batch_size, ntokens]         | |
| | +--------------------------|-------------+               | |
| |                            |                             | |
| |                            v                             | |
| | +--------------------------------------------+           | |
| | |        Backpropagation and Optimization     |           | |
| | |                                            |           | |
| | | - Compute gradients                        |           | |
| | | - Clip gradients (BPTT)                    |           | |
| | | - Update weights                           |           | |
| | +--------------------------------------------+           | |
| |                                                          | |
| +----------------------------------------------------------+ |
|                                                              |
+--------------------------------------------------------------+

