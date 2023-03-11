import { Graphistry } from "@graphistry/client-api-react";

const LOCAL_DEV = {
  graphistryHost: "http://0.0.0.0:3000",
  play: 0,
  session: "cycle",
};

const IFRAME_STYLE = { height: "100%", width: "100%", border: 0 };

export default function Plot({ dispatch, dataset_id, thoughts, busy }) {
  return (
    <Graphistry
      containerClassName="graphistry-outer"
      dataset={dataset_id}
      iframeStyle={IFRAME_STYLE}
    />
  );
}
