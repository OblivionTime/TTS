import os, shutil, sys, pdb, re

now_dir = os.getcwd()
sys.path.insert(0, now_dir)
import json, yaml, warnings, torch

warnings.filterwarnings("ignore")
torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if (os.path.exists(tmp)):
    for name in os.listdir(tmp):
        if (name == "jieba.cache"): continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass
import site

site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if (site_packages_roots == []): site_packages_roots = ["%s/runtime/Lib/site-packages" % now_dir]
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/tools\n%s/tools/damo_asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError:
            pass
print("初始化成功")