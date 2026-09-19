แผนลดการใช้ RAM / VRAM สำหรับโปรเจกต์ MaleCNS → Go2, G1 และโดรน

เครื่องเป้าหมาย: Windows 11, RAM 24 GB, RTX 4060 Laptop 8 GB VRAM
ขอบเขต: ฝึก encoder + decoder โดยคง topology, weights และพารามิเตอร์ของ MaleCNS core ไว้ หากใช้ PPO จะมี value network / critic ขนาดเล็กที่ฝึกเพิ่มด้วย แต่ไม่ใช่การฝึก core
ตรวจเอกสาร: 13 กันยายน 2026
สถานะ: แผนทดลองและเกณฑ์วัด ไม่ใช่ผล benchmark บนเครื่องของผู้ใช้

ข้อจำกัดที่ต้องแยกก่อน

NVIDIA ระบุขั้นต่ำของ Isaac Sim รุ่นปัจจุบันไว้ที่ RAM 32 GB และ VRAM 16 GB และระบุว่า Isaac Lab training ใช้เพิ่มเติม [1] เครื่องนี้จึงต่ำกว่าสเปกที่รองรับ การ optimize เพิ่มโอกาสให้ workload ขนาดเล็กทำงานได้ แต่ไม่รับประกันว่าจะรัน Isaac + full MaleCNS + RL พร้อมกันได้

Flyhard รายงาน peak allocated GPU memory ประมาณ 3 GB ใน steering pilot ที่มี 165,122 neurons และ 25,563,197 neuron-pair edges แต่เป็น rate-based core ที่อัปเดตสี่ครั้งแล้ว reset state ทุก decision ไม่ใช่ long-horizon LIF locomotion RL และไม่ใช่ตัวเลข RAM ของทั้งเครื่อง [2] ห้ามนำ 3 GB ไปใช้เป็นงบที่รับประกันสำหรับงานเรา

1. ค่าเริ่มต้นและเกณฑ์หยุดเพิ่มขนาด

รายการ

ค่าเริ่มต้นที่เสนอ

หุ่นในแต่ละรัน

หนึ่งประเภท / หนึ่ง seed

หุ่นแรก

Go2 บนพื้นเรียบ

Parallel environments

1 ก่อน แล้วค่อยทดสอบ 2 และ 4

ข้อมูลเข้า

joint states, orientation/angular velocity, contacts, task command; ยังไม่ใช้ภาพ

Rendering

Headless, ไม่มี camera/livestream/video ระหว่างฝึก และปิด viewport ที่ไม่ใช้

Neural training microbatch

1 sequence ก่อน แล้วค่อย 2 และ 4

Gradient sequence length

8 control decisions เป็นจุดทดสอบเริ่มต้น ไม่ใช่ 8 LIF integration steps

Precision เริ่มต้น

FP32 สำหรับ dynamics และ sparse core

DataLoader workers

0

เป้าหมาย RAM ทั้งเครื่อง

ไม่เกินประมาณ 80–85% และเหลือ available memory หลาย GB

เป้าหมาย GPU memory ทั้งอุปกรณ์

ไม่เกินประมาณ 85% หรือราว 6.8 GB ของการ์ด 8 GB

ตัวเลขทั้งหมดเป็น ค่าเริ่มทดลอง / เป้าหมายเผื่อพื้นที่ ไม่ใช่การคาดการณ์ว่า Isaac จะกินเท่าไรจริง ต้องตรวจ peak ตอนโหลดฉาก, rollout, backward, optimizer step และ save checkpoint รวมทั้ง throughput หลัง warm-up

2. จัดเก็บกราฟให้เล็กโดยไม่ตัดวงจร

เริ่มจากข้อมูลการเชื่อมต่อระดับคู่เซลล์ ไม่โหลดภาพ EM, skeleton ทุกเซลล์ หรือ mesh สมองเข้ากระบวนการฝึก

แปลงข้อมูลเป็น CSR sparse matrix เพียงครั้งเดียว เก็บเฉพาะค่าการเชื่อมต่อ, column indices และ row pointers พร้อมเก็บตาราง original neuron ID → compact index ไว้แยกกัน ใช้ float32 สำหรับน้ำหนัก และ integer indices ตามที่ backend รองรับ: int32 เมื่อผ่านการทดสอบจริง หรือ int64 เป็นทางเลือก [3]

จากขนาดกราฟ pilot ของ Flyhard คำนวณเองได้ว่า:

Dense FP32 ขนาด 165,122 × 165,122 ต้องใช้ประมาณ 101.57 GiB เฉพาะเมทริกซ์เดียว

CSR แบบ FP32 + int32 indices ใช้ประมาณ 196 MiB; ถ้า indices เป็น int64 ประมาณ 294 MiB

หากเก็บ transpose ใน CSR อีกชุดสำหรับ backward จะรวมประมาณ 391–588 MiB

ตัวเลข CSR นี้เป็นเฉพาะ numeric arrays ไม่รวม temporary buffers, states, autograd, optimizer, framework หรือ simulator

ใช้กราฟชุดเดียวร่วมกับสถานะหลาย environment: W @ H โดย H มีหนึ่งคอลัมน์ต่อ environment ไม่คัดลอก W ตามจำนวน environment และไม่ทำ tensor messages ขนาดทุก edge × ทุก environment × ทุก timestep ค้างไว้

สำหรับ frozen W, backward ของ Y = W @ H ต้องคำนวณ gradient ของ H เป็น W.T @ dY แต่ไม่ต้องคำนวณ gradient ของ W ทดสอบ sparse path กับ dense reference บนกราฟจิ๋วก่อนเสมอ ถ้า backend ยังสร้าง intermediate ใหญ่ ค่อยเขียน custom backward เฉพาะ input gradient ห้ามแอบตัด edges เพื่อให้พอดีโดยไม่บันทึกการเปลี่ยนโมเดล [2,3]

เตรียมกราฟและ randomized controls ทีละตัวใน process แยกจาก Isaac เก็บกราฟสำเร็จไว้บน SSD แล้วปิด process เตรียมข้อมูล หลีกเลี่ยงเก็บทั้ง CSV DataFrame, Python list, COO และ CSR หลายสำเนาพร้อมกัน

3. ฝึก encoder + decoder โดยไม่เผลอตัด gradient

ตั้ง core parameters เป็น requires_grad=False และให้ optimizer รับเฉพาะ encoder, decoder และ critic ที่ใช้จริง ไม่ใส่ graph values เป็น trainable Parameters [4]

ห้ามครอบ core ด้วย torch.no_grad() ในช่วงคำนวณ loss ถ้าต้องการฝึก encoder ผ่านมัน เพราะ gradient ต้องวิ่งกลับผ่าน core ไปถึง encoder การ freeze parameters กับการหยุด autograd เป็นคนละเรื่อง [4]

แบ่งวงจรการฝึกเป็นสองช่วง:

เก็บ rollout: ไม่เก็บ autograd graph; บันทึก observations, actions, rewards, done masks, old log-probabilities และข้อมูลเริ่ม sequence ที่จำเป็นลง CPU buffer ขนาดจำกัด

อัปเดตโมเดล: นำ sequence สั้นกลับมาคำนวณ encoder → core → decoder ใหม่ แล้ว backward; ไม่เก็บ graph ของทั้ง episode

ใช้ truncated BPTT เป็นวิธีประมาณเริ่มต้น: detach gradient history เมื่อข้ามขอบ sequence แต่ คงค่า neural state ไว้ในการจำลอง ไม่ reset state ทุก control step เพียงเพื่อประหยัดหน่วยความจำ ใช้ recurrent-aware PPO batching และ episode masks ให้ถูกต้อง หากใช้ burn-in ให้คำนวณด้วย parameters ปัจจุบัน และทดสอบ sensitivity ของผลต่อความยาว sequence/burn-in

เริ่ม microbatch = 1 และ gradient sequence = 8 control decisions แล้วลอง 16/32 เมื่อพื้นที่พอ ตัวเลขนี้ต้องแยกจากจำนวน integration substeps ภายใน LIF การลด simulation timestep resolution หรือเปลี่ยน time constant เพื่อให้เร็วขึ้นคือการเปลี่ยน dynamics ไม่ใช่ optimization ที่เทียบเท่าเดิม

ใช้ activation checkpointing เมื่อ memory ของ activations เป็นปัญหาจริง: use_reentrant=False, ส่ง state เป็น tensor arguments ชัดเจน, หลีกเลี่ยงแก้ global state แบบ in-place และตรวจ randomness ให้การคำนวณซ้ำตรงกัน วิธีนี้แลก compute เพิ่มกับ memory ลด [5]

Gradient accumulation ช่วยสะสม gradients จาก microbatch โดยไม่เก็บ computation graphs ไว้พร้อมกัน แต่ไม่ได้ลด memory ของ simulator หรือ rollout buffer และไม่ได้ทำให้เก็บ environment samples ได้เร็วขึ้น

เริ่ม core/dynamics เป็น FP32 ก่อน ทดลอง mixed precision เฉพาะ encoder/decoder ภายหลังเมื่อ gradient และพฤติกรรมผ่านการตรวจ ไม่ใช้ INT8/FP16 ทั้ง core เป็นค่าเริ่มต้น เพราะผลต่อ firing thresholds และ numerical stability ต้องวัดแยก

หากใช้ LIF แบบ hard spike ต้องใช้วิธีฝึกผ่าน spike ที่กำหนดไว้ เช่น surrogate gradient; การใส่ sparse matrix ลง PyTorch อย่างเดียวไม่ได้ทำให้ threshold กลายเป็น differentiable อย่าสลับไป rate-based core แล้วเรียกว่า LIF โดยไม่แจ้ง

4. ลดภาระ Isaac โดยไม่บิดโจทย์ locomotion

ใช้หนึ่ง scene เรียบและหุ่นหนึ่งประเภทต่อ process ปิด GUI/cameras/video ระหว่าง training ปิด unused viewport ด้วยวิธีที่ตรงกับ Isaac Sim/Isaac Lab release ที่ติดตั้ง การตั้ง headless อย่างเดียวอาจยังมี default viewport work ใน standalone workflow [6]

ไม่โหลดห้องสมจริง แสง/texture จำนวนมาก ROS bridge, lidar หรือ asset ที่ไม่ใช้ รักษา collision geometry, contacts, actuator limits และ physics settings ที่มีผลกับ gait ไว้ อย่าลดความละเอียด physics จนเกิดการลื่นหรือกระเด้งผิดจริงเพียงเพื่อให้เร็ว

เริ่มจาก environment ที่มีใน Isaac Lab [7]:

Go2: Isaac-Velocity-Flat-Unitree-Go2-v0

G1: Isaac-Velocity-Flat-G1-v0

โดรน Crazyflie: Isaac-Quadcopter-Direct-v0

ใช้ environment เดิมเพื่อตรวจ installation/memory/control interface ก่อน ไม่ใช่ถือว่า reward เดิมเป็น emergence experiment การทดลองเดินเทียบคลานของ G1 ต้องตรวจและแก้ posture/contact rewards, termination, action space และ reset distribution โดยเฉพาะ ไม่ใช้การล็อกแขนหรือ controller เดินสำเร็จรูปแล้วสรุปว่า agent เลือกเดินเอง

Pin เวอร์ชัน Isaac Sim, Isaac Lab, Python, PyTorch และ CUDA ที่เข้ากันและผ่านการทดสอบ ไม่อัปเดตแพ็กเกจทั้งหมดแยกกันเพื่อหวังให้เร็วขึ้น ไม่ใช้ flag จากเอกสาร latest โดยไม่ตรวจว่า release ที่ติดตั้งรองรับ

5. RAM 24 GB: อย่าใช้หมดกับข้อมูลประกอบ

ใช้ num_workers=0 ก่อน เพราะ worker processes อาจเพิ่มสำเนาข้อมูลฝั่ง CPU [8] ไม่เปิด notebook training, viewer, graph preprocessing และ Isaac หลายตัวพร้อมกัน เก็บ dataset/rollout เป็น numeric arrays และอ่านเป็นชิ้นจาก SSD ด้วย memory mapping เมื่อเหมาะสม [9]

ไม่ย้าย activations ทั้งหมดจาก GPU ไป CPU โดยอัตโนมัติ เพราะแก้ VRAM เต็มแต่ไปทำให้ RAM 24 GB เต็มแทน ต้องกำหนด buffer cap ชัดเจน เก็บเฉพาะ sequence ที่จำเป็นและทิ้ง tensors หลังจบ update

Checkpoint ให้แยกกราฟคงที่หนึ่งไฟล์กับ encoder/decoder/critic/optimizer/normalizers/config/RNG state เก็บ graph checksum และ input/output population IDs ไว้ด้วย เพื่อให้โหลดกลับมาแล้วเป็นวงจรเดิม ไม่บันทึกกราฟเต็มซ้ำทุก checkpoint

เปิด Windows page file แบบ system-managed บน SSD ที่มีพื้นที่ว่างพอเพื่อรองรับ commit spikes แต่ไม่ถือว่า page file แทน RAM จริงได้ ถ้ามี paging ต่อเนื่องและ throughput ตก ให้ลด workload [10]

6. บันทึก neural activity แบบที่ยังทำวิจัยได้

อย่าเก็บ activation ทุกเซลล์ทุก timestep ลง Python list ระหว่าง train

ตัวอย่างคำนวณ: 165,122 เซลล์ × 50 samples/s × 600 s × 4 bytes ≈ 18.45 GiB สำหรับตัวแปร FP32 เพียงตัวเดียวและ environment เดียว ถ้าเก็บทั้ง voltage, synaptic state และตัวแปรอื่น ปริมาณยิ่งเพิ่ม

ระหว่าง training เก็บ reward, body state และสถิติกลุ่มเซลล์ เช่น mean/variance/activity counts โดยคำนวณสรุปเป็นช่วง ๆ ส่วนการวิเคราะห์ละเอียดให้โหลด checkpoint แล้วรัน evaluation episode แยกต่างหาก เก็บทุกเซลล์เฉพาะช่วงที่วางแผนไว้ เขียนเป็น chunks ลง SSD ด้วย buffer จำกัดและ detach ก่อนเก็บ

สำหรับโมเดล spiking ให้บันทึก spike counts/events ที่ temporal resolution เหมาะสม ไม่สุ่มอ่าน voltage ที่ 20 Hz แล้วอ้างว่าเห็น spikes ครบ หากมี noise ต้องเก็บ seed/state ที่จำเป็นและตรวจการ replay แทนการสมมุติว่าเหมือนเดิมเสมอ

7. ลำดับทดสอบที่แยกสาเหตุได้

A — Isaac อย่างเดียว: Compatibility Checker → empty headless app → Go2 หนึ่งตัวกับ policy เล็กหรือคำสั่งทดสอบ วัด RAM/VRAM หลัง warm-up และขณะ save/record

B — Neural model อย่างเดียว: ปิด Isaac; โหลด full sparse core, encoder/decoder, microbatch 1 และทดสอบ forward/backward จริง วัดว่า encoder/decoder gradients finite และไม่เป็นศูนย์ทั้งหมด ขณะที่ core ไม่เปลี่ยน ทดสอบหลาย sequence lengths

C — รวมระบบ: หนึ่ง Go2, ไม่มีภาพ, หนึ่ง environment, gradient sequence สั้น เมื่อใช้ memory ต่ำกว่าเกณฑ์และไม่เพิ่มต่อเนื่อง จึงลอง 2 และ 4 environments ทีละระดับ ไม่เปิดทั้งสามร่างพร้อมกัน

D — ขยายงาน: ทำ randomized controls, seeds และ embodiments ทีละรัน ใช้สเปกการฝึกเหมือนกันในคู่เปรียบเทียบ และรายงานข้อจำกัดของ truncated gradients

หาก A ไม่ผ่าน แม้ scene เล็กแล้ว การลดขนาด neural core ไม่ช่วยแก้ base cost ของ Isaac ต้องใช้เครื่องอื่นสำหรับ simulator หรือประเมิน simulator ที่เบากว่าเป็นแผนสำรองอย่างเปิดเผย

หาก A กับ B ผ่านแยกกันแต่ C ไม่ผ่าน ลอง workflow แบบสลับ process: เก็บ rollout ภายใต้ current policy โดยไม่ทำ backward → ปิด process Isaac → อัปเดต policy จาก rollout นั้น → เปิด simulator เก็บ rollout ใหม่ การ restart มี overhead และ PPO ห้ามใช้ rollout เก่าวนไม่จำกัดเมื่อ policy เปลี่ยน หาก core เป็น recurrent ต้องรักษา episode boundaries และ reconstruction ของ state ให้ถูกต้อง

หาก B ไม่ผ่านเต็มกราฟ ให้เลือก locomotor subnetwork ด้วยเกณฑ์ทางกายวิภาคล่วงหน้า แล้วทำ matched controls บน subnetwork เดียวกัน ชัดเจนว่านี่คือ เปลี่ยนขอบเขตโมเดล ไม่ใช่การบีบอัดที่คง full MaleCNS ทุกอย่าง

8. วัดอะไรและใช้อะไรตัดสิน

สคริปต์ memory_watch.py ที่แนบเป็นตัวอ่าน RAM ทั้งเครื่องและ GPU memory ผ่าน nvidia-smi ทุกสองวินาที บันทึก CSV และเตือนที่ 85% โดยไม่แก้ settings และไม่หยุด process ให้อัตโนมัติ [11]

python -m pip install psutil
python memory_watch.py --output go2_memory.csv

ดู GPU Dedicated memory และ RAM/Available ใน Windows Task Manager ควบคู่กัน บน Windows/driver บางชุด GPU query อาจรายงาน N/A ให้ใช้ Task Manager แทน

ใน process training ให้เก็บ torch.cuda.max_memory_allocated() และ torch.cuda.max_memory_reserved() หลัง synchronize ในแต่ละขั้น ทั้งสองค่าไม่รวม memory ที่ Isaac/driver จัดสรรนอก PyTorch และ empty_cache() ไม่ได้ลบ live tensors หรือทำให้โมเดลที่ใหญ่เกินไปพอดีทันที [12]

เกณฑ์ผ่านต้องมีทั้ง memory เหลือ, ไม่เกิด NaN/OOM, encoder ได้ gradient, core weights ไม่เปลี่ยน, throughput ใช้งานได้ และการจำลองยังรักษา physics/dynamics ตามที่ประกาศ การรันได้อย่างเดียวไม่เท่ากับฝึกให้ locomotion สำเร็จ

การทดสอบไฟล์แนบ: ตรวจ syntax และทดสอบ RAM logging/กรณีไม่มี nvidia-smi ในสภาพแวดล้อม Linux แบบ CPU เท่านั้น ยังไม่ได้ทดสอบบน Windows 11, RTX 4060, MaleCNS หรือ Isaac ของผู้ใช้

แหล่งอ้างอิง

[1] NVIDIA Isaac Sim requirements: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html

[2] Flyhard pilot 2026-09-09, รายงานของผู้พัฒนา: https://github.com/MarkUnthank/flyhard/blob/main/docs/pilot-2026-09-09.md

[3] PyTorch sparse matrix multiplication: https://docs.pytorch.org/docs/2.14/generated/torch.sparse.mm.html

[4] PyTorch autograd / freezing parameters: https://docs.pytorch.org/docs/2.14/notes/autograd.html

[5] PyTorch activation checkpointing: https://docs.pytorch.org/docs/2.14/checkpoint.html

[6] NVIDIA performance optimization handbook: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html

[7] Isaac Lab available environments: https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html

[8] PyTorch DataLoader: https://docs.pytorch.org/docs/2.14/data.html

[9] NumPy memory-mapped arrays: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

[10] Microsoft page file documentation: https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/introduction-to-the-page-file

[11] NVIDIA SMI: https://docs.nvidia.com/deploy/nvidia-smi/index.html

[12] PyTorch CUDA memory management: https://docs.pytorch.org/docs/2.14/notes/cuda.html