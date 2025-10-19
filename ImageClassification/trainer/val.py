import torch
import utils
import numpy as np
from imagenet import get_x_y_from_data_dict


def validate(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(val_loader):
            image, target = get_x_y_from_data_dict(data, device)
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

        print("valid_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        for i, (image, target) in enumerate(val_loader):
            image = image.cuda()
            target = target.cuda()

            # compute output
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

        print("valid_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


def bd_validate(val_loader, model, criterion, args):
    """
    Run evaluation
    """

    def generate_mask(image):
        h, w = image.size(2), image.size(3)
        mask = torch.ones_like(image)
        # set 4x4 zeros at random location
        for i in range(image.size(0)):
            x = torch.randint(0, w - 4, (1,))
            y = torch.randint(0, h - 4, (1,))
            mask[i, :, y : y + 4, x : x + 4] = 0.0
        return mask

    losses = utils.AverageMeter()
    losses_bd = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top1_bd = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(val_loader):
            image, target = get_x_y_from_data_dict(data, device)
            num_sample = image.size(0)

            image_bd = image.clone()
            image_bd[:, :, :4, :4] = 0.0  # TODO: larger patch for imagenet
            # image_bd[:, :, :3, :3] = 0.
            # image_bd[:, :, :3, -3:] = 0.
            # image_bd[:, :, -3:, :3] = 0.
            # image_bd[:, :, -3:, -3:] = 0.
            image = torch.cat((image, image_bd), 0)

            with torch.no_grad():
                output = model(image)
                loss = criterion(output[:num_sample], target)
                loss_bd = criterion(output[num_sample:], target)

            output = output.float()
            loss = loss.float()
            loss_bd = loss_bd.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output[:num_sample].data, target)[0]
            prec2 = utils.accuracy(output[num_sample:].data, target)[0]
            losses.update(loss.item(), image.size(0))
            losses_bd.update(loss_bd.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            top1_bd.update(prec2.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Loss_bd {loss_bd.val:.4f} ({loss_bd.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "BD Accuracy {top1_bd.val:.3f} ({top1_bd.avg:.3f})".format(
                        i,
                        len(val_loader),
                        loss=losses,
                        loss_bd=losses_bd,
                        top1=top1,
                        top1_bd=top1_bd,
                    )
                )

        print(
            "valid_accuracy {top1.avg:.3f}  valid_bd_accuracy {top1_bd.avg:.3f}".format(
                top1=top1, top1_bd=top1_bd
            )
        )
    else:
        for i, (image, target) in enumerate(val_loader):
            image = image.cuda()
            target = target.cuda()
            num_sample = image.size(0)

            image_bd = image.clone()

            if args.bd_pix == 36:
                image_bd[:, :, :3, :3] = 0.0
                image_bd[:, :, :3, -3:] = 0.0
                image_bd[:, :, -3:, :3] = 0.0
                image_bd[:, :, -3:, -3:] = 0.0
            else:
                patch_size = int(np.sqrt(args.bd_pix))
                image_bd[:, :, :patch_size, :patch_size] = 0.0  # fixed location
            # mask_bd = generate_mask(image_bd)  # random location
            # image_bd = image_bd * mask_bd

            image = torch.cat((image, image_bd), 0)

            with torch.no_grad():
                output = model(image)
                loss = criterion(output[:num_sample], target)
                loss_bd = criterion(output[num_sample:], target)

            output = output.float()
            loss = loss.float()
            loss_bd = loss_bd.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output[:num_sample].data, target)[0]
            prec2 = utils.accuracy(output[num_sample:].data, target)[0]
            losses.update(loss.item(), image.size(0))
            losses_bd.update(loss_bd.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            top1_bd.update(prec2.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Loss_bd {loss_bd.val:.4f} ({loss_bd.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "BD Accuracy {top1_bd.val:.3f} ({top1_bd.avg:.3f})".format(
                        i,
                        len(val_loader),
                        loss=losses,
                        loss_bd=losses_bd,
                        top1=top1,
                        top1_bd=top1_bd,
                    )
                )

        print(
            "valid_accuracy {top1.avg:.3f}  valid_bd_accuracy {top1_bd.avg:.3f}".format(
                top1=top1, top1_bd=top1_bd
            )
        )

    return top1.avg
