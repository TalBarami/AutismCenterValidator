set PATH=%PATH%;C:\Users\User\anaconda3;C:\Users\User\anaconda3\Scripts
set PYTHONPATH=%PATH%;E:\AutismCenterValidator;E:\AutismCenterValidator\src
call activate annotate
cd "E:/AutismCenterValidator"
python "src/validator/annotate_smms/smm_annotator.py"

