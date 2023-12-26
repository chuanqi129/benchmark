model_list = [
"BERT_pytorch",
"Background_Matting",        
"DALLE2_pytorch",
"LearningToPaint",
"Super_SloMo",
"alexnet",
#"attention_is_all_you_need_pytorch", remove from latest officail torchbench
"dcgan",
"demucs",
"densenet121",
"dlrm",
"drq",
"fambench_xlmr",
"fastNLP_Bert",
"functorch_dp_cifar10",
"functorch_maml_omniglot",
"hf_Albert",
"hf_Bart",
"hf_Bert",
"hf_Bert_large",
"hf_BigBird",
"hf_DistilBert",
"hf_GPT2",
"hf_GPT2_large",
"hf_Longformer",
"hf_Reformer",
"hf_T5",
"hf_T5_base",
"hf_T5_large",
"lennard_jones",
"maml",
"maml_omniglot",
"mnasnet1_0",
"mobilenet_v2",
"mobilenet_v2_quantized_qat",
"mobilenet_v3_large",
"moco",
"nvidia_deeprecommender",
"opacus_cifar10",
"phlippe_densenet",
"phlippe_resnet",
"pyhpc_equation_of_state",
"pyhpc_isoneutral_mixing",
"pyhpc_turbulent_kinetic_energy",
"pytorch_CycleGAN_and_pix2pix",
"pytorch_stargan",
#"pytorch_struct", remove from latest officail torchbench
"pytorch_unet",
"resnet152",
"resnet18",
"resnet50",
"resnet50_quantized_qat",
"resnext50_32x4d",
"shufflenet_v2_x1_0",
"soft_actor_critic",
"speech_transformer",
"squeezenet1_1",
"tacotron2",
"timm_efficientdet",
"timm_efficientnet",
"timm_nfnet",
"timm_regnet",
"timm_resnest",
"timm_vision_transformer",
"timm_vision_transformer_large",
"timm_vovnet",
"torchrec_dlrm",
"tts_angular",
"vgg16",
"vision_maskrcnn",
"yolov3",
"detectron2_fasterrcnn_r_101_c4",
"detectron2_fasterrcnn_r_101_dc5",
"detectron2_fasterrcnn_r_101_fpn",
"detectron2_fasterrcnn_r_50_c4",
"detectron2_fasterrcnn_r_50_dc5",
"detectron2_fasterrcnn_r_50_fpn",
"detectron2_fcos_r_50_fpn",
"detectron2_maskrcnn",
"detectron2_maskrcnn_r_101_c4",
"detectron2_maskrcnn_r_101_fpn",
"detectron2_maskrcnn_r_50_c4",
"detectron2_maskrcnn_r_50_fpn",
"doctr_det_predictor",
"doctr_reco_predictor"
]
recheck_list = [
"detectron2_maskrcnn_r_101_fpn"
        ]

import subprocess
import argparse

def run(args):
    if args.jit:
        mode_list = ["eval"]
        model_list_tmp = []
        for idx, model in enumerate(model_list):
            if "hf_" not in model:
                model_list_tmp.append(model)
            else:
                print("%s don't support jit" % model)
    else:
        mode_list = ["train", "eval"]
        model_list_tmp = model_list
    
    #for model in recheck_list:
    for model in model_list_tmp:
        for mode in mode_list:
            cmd = "python run.py %s -d %s -t %s --precision %s" % (model, args.device, mode, args.precision)
            if args.jit:
                cmd = cmd + " --backend torchscript"
            if args.optimize:
                cmd = cmd + " --optimize"
            print("===========================")
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.wait()
            output = process.stdout.read()
            error = process.stderr.read()

            output = output.decode(encoding="gbk")
            try:
                error = error.decode(encoding="gbk")
            except Exception:
                print("skip erroe decode gbk")
        
            result = "%s\n output: %s \n error:%s \n" % (cmd, str(output), str(error))
            print(result)
            
            gpu_time = "error"
            cpu_time = "error"
            bs = "error"
            throughput = "error"

            if "batch size" in result:
                bs = result.split("batch size")[-1].strip()
                bs = bs.split("and precision")[0].strip()
            
            if str(error) != '' and ("Error" in str(error)):
                if "xe2" in str(error):
                    throughput = str(error).split('\\n')[-2]
                else:
                    throughput = str(error).splitlines()[-1]

            if str(output) != '' and ("Time" in str(output)):
                gpu_time_key = "GPU Time per batch:"
                cpu_time_key = "CPU Wall Time per batch:"
                gpu_time = output.split(gpu_time_key)[-1].split("milliseconds")[0].strip()
                cpu_time = output.split(cpu_time_key)[-1].split("milliseconds")[0].strip()
                # bs / latency
                throughput = int(bs)/(float(cpu_time)/1000)
        
            # Summary: resnet50 train fp32 bs throughput
            print("Summary: %s %s %s %s %s %s %s" % (model, mode, args.precision, bs, gpu_time, cpu_time, throughput))

def main():
    parser = argparse.ArgumentParser(description=f"Models results parser")
    parser.add_argument('--device', '-d', default="xpu", help='device type, should be xpu, cpu, ipex_xpu, cuda')
    parser.add_argument('--precision', default="fp32", help='dtype, should be fp32, amp')
    parser.add_argument('--optimize', default="False", help='True or False, enable optimize or not')
    parser.add_argument('--jit', default="False", help='True or False, enable jit or not')
    args = parser.parse_args()
    args.optimize = False if args.optimize == "False" else True
    args.jit = False if args.jit == "False" else True
    
    run(args)

if __name__ == "__main__":
    main()
