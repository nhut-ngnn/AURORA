import torch
import torch.nn as nn
import torch.nn.functional as F

class UCHLoss(nn.Module):
    def __init__(
        self,
        lambda_hallucination=1.0,
        lambda_distill=1.0,
        lambda_kd=0.5,
        temperature=2.0,
        strict_teacher=True,
    ):
        super().__init__()
        self.lambda_hallucination = lambda_hallucination
        self.lambda_distill = lambda_distill
        self.lambda_kd = lambda_kd
        self.temperature = temperature
        self.strict_teacher = strict_teacher

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_out, labels, teacher_out=None):
        logits_s = student_out["logits_student"]
        l_cls = self.ce_loss(logits_s, labels)

        if teacher_out is None:
            return l_cls, {"l_total": l_cls.item(), "l_cls": l_cls.item()}

        z_clean = teacher_out.get("z_clean_text", None)
        rep_t = teacher_out.get("teacher_rep", None)
        logits_t = teacher_out.get("logits_teacher", None)

        if self.strict_teacher:
            if z_clean is None or rep_t is None or logits_t is None:
                raise KeyError(
                    "Teacher outputs missing. Need keys: z_clean_text, teacher_rep, logits_teacher. "
                    "Did you pass clean_text with mode='both'?"
                )
        else:
            if z_clean is None:
                z_clean = teacher_out.get("z_asr_text", None)
            if rep_t is None:
                rep_t = teacher_out.get("student_rep", None)
            if logits_t is None:
                logits_t = teacher_out.get("logits_student", None)

        z_hallu = student_out["z_hallucinated"]
        if z_clean is None:
            l_hallu = torch.zeros((), device=logits_s.device)
        else:
            z_clean = z_clean.detach()
            l_hallu = 1.0 - F.cosine_similarity(z_hallu, z_clean, dim=-1).mean()


        rep_s = student_out["student_rep"]
        if rep_t is None:
            l_distill = torch.zeros((), device=logits_s.device)
        else:
            rep_t = rep_t.detach()
            l_distill = self.mse_loss(rep_s, rep_t)

        if logits_t is None:
            l_kd = torch.zeros((), device=logits_s.device)
        else:
            T = self.temperature
            logits_t = logits_t.detach()
            p_t = F.softmax(logits_t / T, dim=-1)
            log_p_s = F.log_softmax(logits_s / T, dim=-1)
            l_kd = self.kl_loss(log_p_s, p_t) * (T * T)

        l_total = (
            l_cls
            + self.lambda_hallucination * l_hallu
            + self.lambda_distill * l_distill
            + self.lambda_kd * l_kd
        )

        loss_dict = {
            "l_total": float(l_total.detach().cpu()),
            "l_cls": float(l_cls.detach().cpu()),
            "l_hallu": float(l_hallu.detach().cpu()),
            "l_distill": float(l_distill.detach().cpu()),
            "l_kd": float(l_kd.detach().cpu()),
            "alpha_mean": float(student_out["alpha"].mean().detach().cpu()),
        }
        return l_total, loss_dict


class UCHLossNoHallucination(nn.Module):
    def __init__(
        self,
        lambda_distill=1.0,
        lambda_kd=1.0,
        temperature=2.0,
        strict_teacher=True,
    ):
        super().__init__()
        self.lambda_distill = lambda_distill
        self.lambda_kd = lambda_kd
        self.temperature = temperature
        self.strict_teacher = strict_teacher

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_out, labels, teacher_out=None):
        logits_s = student_out["logits_student"]
        l_cls = self.ce_loss(logits_s, labels)

        if teacher_out is None:
            return l_cls, {"l_total": l_cls.item(), "l_cls": l_cls.item(), "l_hallu": 0.0}

        rep_t = teacher_out.get("teacher_rep", None)
        logits_t = teacher_out.get("logits_teacher", None)

        if self.strict_teacher:
            if rep_t is None or logits_t is None:
                raise KeyError(
                    "Teacher outputs missing. Need keys: teacher_rep, logits_teacher. "
                    "Did you pass clean_text with mode='both'?"
                )
        else:
            if rep_t is None:
                rep_t = teacher_out.get("student_rep", None)
            if logits_t is None:
                logits_t = teacher_out.get("logits_student", None)

        rep_s = student_out["student_rep"]
        if rep_t is None:
            l_distill = torch.zeros((), device=logits_s.device)
        else:
            rep_t = rep_t.detach()
            l_distill = self.mse_loss(rep_s, rep_t)

        if logits_t is None:
            l_kd = torch.zeros((), device=logits_s.device)
        else:
            T = self.temperature
            logits_t = logits_t.detach()
            p_t = F.softmax(logits_t / T, dim=-1)
            log_p_s = F.log_softmax(logits_s / T, dim=-1)
            l_kd = self.kl_loss(log_p_s, p_t) * (T * T)

        l_total = l_cls + self.lambda_distill * l_distill + self.lambda_kd * l_kd

        loss_dict = {
            "l_total": float(l_total.detach().cpu()),
            "l_cls": float(l_cls.detach().cpu()),
            "l_hallu": 0.0,
            "l_distill": float(l_distill.detach().cpu()),
            "l_kd": float(l_kd.detach().cpu()),
            "alpha_mean": float(student_out["alpha"].mean().detach().cpu()),
        }
        return l_total, loss_dict


class UCHLossCEDistill(nn.Module):
    def __init__(self, lambda_distill=1.0, strict_teacher=True):
        super().__init__()
        self.lambda_distill = lambda_distill
        self.strict_teacher = strict_teacher
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_out, labels, teacher_out=None):
        logits_s = student_out["logits_student"]
        l_cls = self.ce_loss(logits_s, labels)

        if teacher_out is None:
            return l_cls, {
                "l_total": float(l_cls.detach().cpu()),
                "l_cls": float(l_cls.detach().cpu()),
                "l_hallu": 0.0,
                "l_distill": 0.0,
                "l_kd": 0.0,
                "alpha_mean": float(student_out["alpha"].mean().detach().cpu()) if "alpha" in student_out else 0.0,
            }

        rep_t = teacher_out.get("teacher_rep", None)
        if self.strict_teacher and rep_t is None:
            raise KeyError(
                "Teacher outputs missing. Need key: teacher_rep. "
                "Did you pass clean_text with mode='both'?"
            )
        if rep_t is None:
            rep_t = teacher_out.get("student_rep", None)

        rep_s = student_out["student_rep"]
        if rep_t is None:
            l_distill = torch.zeros((), device=logits_s.device)
        else:
            rep_t = rep_t.detach()
            l_distill = self.mse_loss(rep_s, rep_t)

        l_total = l_cls + self.lambda_distill * l_distill

        return l_total, {
            "l_total": float(l_total.detach().cpu()),
            "l_cls": float(l_cls.detach().cpu()),
            "l_hallu": 0.0,
            "l_distill": float(l_distill.detach().cpu()),
            "l_kd": 0.0,
            "alpha_mean": float(student_out["alpha"].mean().detach().cpu()) if "alpha" in student_out else 0.0,
        }


class UCHLossCEHallucination(nn.Module):
    def __init__(self, lambda_hallucination=1.0, strict_teacher=True):
        super().__init__()
        self.lambda_hallucination = lambda_hallucination
        self.strict_teacher = strict_teacher
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_out, labels, teacher_out=None):
        logits_s = student_out["logits_student"]
        l_cls = self.ce_loss(logits_s, labels)

        if teacher_out is None:
            return l_cls, {
                "l_total": float(l_cls.detach().cpu()),
                "l_cls": float(l_cls.detach().cpu()),
                "l_hallu": 0.0,
                "l_distill": 0.0,
                "l_kd": 0.0,
                "alpha_mean": float(student_out["alpha"].mean().detach().cpu()) if "alpha" in student_out else 0.0,
            }

        z_clean = teacher_out.get("z_clean_text", None)
        if self.strict_teacher and z_clean is None:
            raise KeyError(
                "Teacher outputs missing. Need key: z_clean_text. "
                "Did you pass clean_text with mode='both'?"
            )
        if z_clean is None:
            z_clean = teacher_out.get("z_asr_text", None)

        z_hallu = student_out["z_hallucinated"]
        if z_clean is None:
            l_hallu = torch.zeros((), device=logits_s.device)
        else:
            z_clean = z_clean.detach()
            l_hallu = 1.0 - F.cosine_similarity(z_hallu, z_clean, dim=-1).mean()

        l_total = l_cls + self.lambda_hallucination * l_hallu

        return l_total, {
            "l_total": float(l_total.detach().cpu()),
            "l_cls": float(l_cls.detach().cpu()),
            "l_hallu": float(l_hallu.detach().cpu()),
            "l_distill": 0.0,
            "l_kd": 0.0,
            "alpha_mean": float(student_out["alpha"].mean().detach().cpu()) if "alpha" in student_out else 0.0,
        }


class UCHLossCEKD(nn.Module):
    def __init__(self, lambda_kd=1.0, temperature=2.0, strict_teacher=True):
        super().__init__()
        self.lambda_kd = lambda_kd
        self.temperature = temperature
        self.strict_teacher = strict_teacher
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_out, labels, teacher_out=None):
        logits_s = student_out["logits_student"]
        l_cls = self.ce_loss(logits_s, labels)

        if teacher_out is None:
            return l_cls, {
                "l_total": float(l_cls.detach().cpu()),
                "l_cls": float(l_cls.detach().cpu()),
                "l_hallu": 0.0,
                "l_distill": 0.0,
                "l_kd": 0.0,
                "alpha_mean": float(student_out["alpha"].mean().detach().cpu()) if "alpha" in student_out else 0.0,
            }

        logits_t = teacher_out.get("logits_teacher", None)
        if self.strict_teacher and logits_t is None:
            raise KeyError(
                "Teacher outputs missing. Need key: logits_teacher. "
                "Did you pass clean_text with mode='both'?"
            )
        if logits_t is None:
            logits_t = teacher_out.get("logits_student", None)

        if logits_t is None:
            l_kd = torch.zeros((), device=logits_s.device)
        else:
            T = self.temperature
            logits_t = logits_t.detach()
            p_t = F.softmax(logits_t / T, dim=-1)
            log_p_s = F.log_softmax(logits_s / T, dim=-1)
            l_kd = self.kl_loss(log_p_s, p_t) * (T * T)

        l_total = l_cls + self.lambda_kd * l_kd

        return l_total, {
            "l_total": float(l_total.detach().cpu()),
            "l_cls": float(l_cls.detach().cpu()),
            "l_hallu": 0.0,
            "l_distill": 0.0,
            "l_kd": float(l_kd.detach().cpu()),
            "alpha_mean": float(student_out["alpha"].mean().detach().cpu()) if "alpha" in student_out else 0.0,
        }


class UCHLossCEOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_out, labels, teacher_out=None):
        logits_s = student_out["logits_student"]
        l_cls = self.ce_loss(logits_s, labels)
        return l_cls, {
            "l_total": float(l_cls.detach().cpu()),
            "l_cls": float(l_cls.detach().cpu()),
            "l_hallu": 0.0,
            "l_distill": 0.0,
            "l_kd": 0.0,
            "alpha_mean": float(student_out["alpha"].mean().detach().cpu()) if "alpha" in student_out else 0.0,
        }
