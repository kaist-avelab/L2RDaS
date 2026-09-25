import torch


def create_model(opt, cfg):
    if opt.model == "pix2pixHD":
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel

        if cfg.DATASET.object_sample.make_l2r_tensor:
            model = InferenceModel()
        elif opt.isTrain:
            model = Pix2PixHDModel()
        else:
            print("Wrong! Error in making model!")
    else:
        raise ValueError(
            f"Unsupported model '{opt.model}'. This release includes pix2pixHD only."
        )

    model.initialize(opt, cfg)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if (
        opt.isTrain
        and len(opt.gpu_ids)
        and not opt.fp16
        and not cfg.DATASET.object_sample.make_l2r_tensor
    ):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
