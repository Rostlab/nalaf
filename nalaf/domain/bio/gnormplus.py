import difflib
from nalaf.learning.taggers import Tagger
from nalaf.utils.ncbi_utils import GNormPlus
from nalaf.utils.uniprot_utils import Uniprot
from nalaf.structures.data import Entity
from nalaf.utils import PRO_CLASS_ID, ENTREZ_GENE_ID, UNIPROT_ID


class GNormPlusGeneTagger(Tagger):
    """
    Performs tagging for genes with GNormPlus.
    Is able to add normalisations for uniprot as well.

    :type crf_suite: nalaf.learning.crfsuite.CRFSuite
    """

    def __init__(self):
        super().__init__([ENTREZ_GENE_ID, UNIPROT_ID])

    def __find_offset_adjustments(self, s1, s2, start_offset):
        return [(start_offset+i1, j2-j1-i2+i1)
                   for type, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s1, s2).get_opcodes()
                   if type in ['replace', 'insert']]

    def tag(self, dataset, annotated=False, uniprot=False, process_only_abstract=True):
        """
        :type dataset: nalaf.structures.data.Dataset
        :param annotated: if True then saved into annotations otherwise into predicted_annotations
        """
        with GNormPlus() as gnorm:
            for doc_id, doc in dataset.documents.items():
                if process_only_abstract:
                    genes, gnorm_title, gnorm_abstract = gnorm.get_genes_for_pmid(doc_id, postproc=True)

                    if uniprot:
                        with Uniprot() as uprot:
                            list_of_ids = gnorm.uniquify_genes(genes)
                            genes_mapping = uprot.get_uniprotid_for_entrez_geneid(list_of_ids)
                    else:
                        genes_mapping = {}

                    # find the title and the abstract
                    parts = iter(doc.parts.values())
                    title = next(parts)
                    abstract = next(parts)
                    adjustment_offsets = []
                    if title.text != gnorm_title:
                        adjustment_offsets += self.__find_offset_adjustments(title.text, gnorm_title, 0)
                    if abstract.text != gnorm_abstract:
                        adjustment_offsets += self.__find_offset_adjustments(abstract.text, gnorm_abstract, len(gnorm_title))

                    for start, end, text, gene_id in genes:
                        if 0 <= start < end <= len(title.text):
                            part = title
                        else:
                            part = abstract
                            # we have to readjust the offset since GnormPlus provides
                            # offsets for title and abstract together
                            offset = len(title.text) + 1
                            start -= offset
                            end -= offset

                        for adjustment_offset, adjustment in adjustment_offsets:
                            if start > adjustment_offset:
                                start -= adjustment

                        # todo discussion which confidence value for gnormplus because there is no value supplied
                        ann = Entity(class_id=PRO_CLASS_ID, offset=start, text=text, confidence=0.5)
                        try:
                            norm_dict = {
                                ENTREZ_GENE_ID: gene_id,
                                UNIPROT_ID: genes_mapping[gene_id]
                            }
                        except KeyError:
                            norm_dict = {ENTREZ_GENE_ID: gene_id}

                        norm_string = ''  # todo normalized_text (stemming ... ?)
                        ann.normalisation_dict = norm_dict
                        ann.normalized_text = norm_string
                        if annotated:
                            part.annotations.append(ann)
                        else:
                            part.predicted_annotations.append(ann)
                else:
                    # todo this is not used for now anywhere, might need to be re-worked or excluded
                    # genes = gnorm.get_genes_for_text(part.text)
                    pass
