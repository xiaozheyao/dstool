class ImageEncoder:
    def __init__(self, model_name:str = "facebook/dinov2-large", device="gpu"):
        try:
            from transformers import AutoModel, AutoImageProcessor
        except ImportError:
            raise ImportError("Please install transformers to use ImageEncoder.")
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        if torch.cuda.is_available():
            print(f"Using GPU for {model_name} model.")
            self.model.to("cuda")

    def compute_embedding(self, images: List[Image.Image], cls_token_only: bool = True):
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs.to(self.model.device)

            embeddings = self.model(**inputs).last_hidden_state

        if cls_token_only:
            embeddings = embeddings[:, 0, :]
        return embeddings.cpu()

def encode_video_using_image_encoder(
        encoder: ImageEncoder,
        video_path: str,
        batch_size: int = 1024,
        cls_token_only: bool = True
    ):
    try:
        from decord import VideoReader, cpu, gpu
    except ImportError:
        raise ImportError("Please install decord to use encode_video_using_image_encoder.")
    vr = VideoReader(video_path, ctx=cpu(0))
    embeddings = []

    for i in tqdm(range(0, len(vr), batch_size)):

        batch_indices = range(i, min(i + batch_size, len(vr)))
        batch_images = [Image.fromarray(vr[j].asnumpy()).convert("RGB") for j in batch_indices]
        
        embeddings.extend(encoder.compute_embedding(batch_images, cls_token_only=cls_token_only))

    assert len(embeddings) == len(vr), "Number of embeddings does not match number of frames"
    return torch.stack(embeddings)