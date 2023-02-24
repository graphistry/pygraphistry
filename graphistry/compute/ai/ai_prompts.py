import symai as ai
from symai import Prompt, Symbol, Sequence, Stream, Cluster


NEXT = " =>"


class SplunkPrompts(Prompt):
    # https://docs.splunk.com/Documentation/SCS/current/SearchReference/SearchCommandExamples
    def __init__(self) -> Prompt:
        super().__init__(
            [
                "search the main index where sourcetype is access_combined limit 10"
                + NEXT
                + "search index=main sourcetype=access_combined | head 10",
                "Modify the following Splunk query that is useful for identifying malware in FireEye logs to instead use Palo Alto Networks logs:"
                + NEXT
                + "index=palo_alto sourcetype=pan:logs (action=allow OR action=deny) | eval src_ip=src_ip | eval dest_ip=dest_ip | where (src_ip=* OR dest_ip=*) | stats count by src_ip, dest_ip, action | sort - count",
                "Lets try to find out how many errors have occurred on the Buttercup Games website"
                + NEXT
                + "buttercupgames (error OR fail* OR severe)",
                "write a splunk query for the index `redteam_50k` that uses the src and dst information to output a table for events where RED=1. you can use closest matching fields from [src_computer, other, dst_computer, time]"
                + NEXT
                + '| search index="redteam_50k" RED=1 | Table src_computer, dst_computer',
                "get the top 10 most common source IPs in the index `redteam_50k`"
                + NEXT
                + '| search index="redteam_50k" | stats count by src_ip | sort - count | head 10',
                "get fields for the index redteam_50k"
                + NEXT
                + '| search index="redteam_50k" | fieldsummary | table field',
            ]
        )


class Splunk(Symbol):
    def __init__(self, value="", *args, **kwargs) -> None:
        super().__init__(value, *args, **kwargs)

    @ai.few_shot(
        prompt="You are an expert in splunk: Output * as splunk code block",
        examples=SplunkPrompts(),
    )
    def as_splunk(self):
        return self.value


class Journey(Symbol):
    def __init__(self, value="", *args, **kwargs) -> None:
        super().__init__(value, *args, **kwargs)

    def forward(self, value):
        stream = Stream(Sequence())
        sym = Symbol(value)
        res = Symbol(list(stream(sym)))
        expr = Cluster()
        return expr(res)
