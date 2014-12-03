ansible-playbook system.yml -vv --tags fast --skip-tags provision,staging-slack -i hosts -l prod
