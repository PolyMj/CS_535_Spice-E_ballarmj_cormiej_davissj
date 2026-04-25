import argparse
import eval
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Directory containing all results from inference.")
    parser.add_argument("-g", "--guidance", required=True, help="Path to 3D model used for guidance.")
    parser.add_argument("-p", "--prompt", required=False, help="The text prompt")
    parser.add_argument("-o", "--output_json", required=False, help="The outputted json file containing metrics. ")
    args = parser.parse_args()

    if (args.input[-1] in ['\\', '/']):
        args.input = args.input[0:-1]
    
    if (args.prompt is None or len(args.prompt) < 1):
        args.prompt = os.path.basename(args.input)
        print(args.input)
    
    if (args.output_json is None):
        args.output_json = os.path.join("metrics", os.path.basename(args.guidance))
        args.output_json = os.path.join(args.output_json, args.prompt)
        args.output_json += ".json"
        

    args.input = os.path.join(args.input, "output/")

    eval.create_metrics(
        images_dir=args.input,
        prompt=args.prompt,
        guidance=args.guidance,
        result=os.path.join(args.input, "output.ply"),
        source_image=os.path.join(args.input, "output.pt"),
        output_json=args.output_json
    )



if __name__ == "__main__":
    main()