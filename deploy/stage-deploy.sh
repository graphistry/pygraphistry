ansible-playbook system.yml -vv --tags fast --skip-tags provision,prod-slack -i hosts -l staging
