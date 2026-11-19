import ca.uhn.fhir.context.FhirContext;
import ca.uhn.fhir.validation.*;
import ca.uhn.fhir.context.support.DefaultProfileValidationSupport;
import org.hl7.fhir.common.hapi.validation.support.*;
import org.hl7.fhir.common.hapi.validation.validator.FhirInstanceValidator;
import java.nio.file.*;
import java.util.*;

/** Validates FHIR R4 JSON resources/bundles with the HAPI instance validator
 *  against the base R4 StructureDefinitions (offline; no terminology server). */
public class Validate {
  public static void main(String[] a) throws Exception {
    FhirContext ctx = FhirContext.forR4();
    ValidationSupportChain chain = new ValidationSupportChain(
        new DefaultProfileValidationSupport(ctx),
        new InMemoryTerminologyServerValidationSupport(ctx),
        new CommonCodeSystemsTerminologyService(ctx));
    FhirInstanceValidator inst = new FhirInstanceValidator(chain);
    inst.setNoTerminologyChecks(true);
    FhirValidator v = ctx.newValidator().registerValidatorModule(inst);
    List<String> files = new ArrayList<>();
    for (String s : a) {
      Path p = Paths.get(s);
      if (Files.isDirectory(p)) try (var st = Files.list(p)) { st.filter(x->x.toString().endsWith(".json")).sorted().forEach(x->files.add(x.toString())); }
      else files.add(s);
    }
    for (String f : files) {
      String txt = Files.readString(Paths.get(f));
      long t0 = System.nanoTime();
      ValidationResult r = v.validateWithResult(txt);
      long ms = (System.nanoTime()-t0)/1000000;
      int err=0, warn=0; StringBuilder msgs = new StringBuilder();
      for (SingleValidationMessage m : r.getMessages()) {
        if (m.getSeverity()==ResultSeverityEnum.ERROR || m.getSeverity()==ResultSeverityEnum.FATAL) {
          err++; if (System.getenv("ALLMSG")!=null) System.out.println("MSG\t"+f+"\t"+m.getLocationString()+" :: "+m.getMessage().replace("\n"," ")); if (err<=5) msgs.append(m.getLocationString()).append(" :: ").append(m.getMessage().replace("\"","'").replace("\t"," ").replace("\n"," ")).append(" || ");
        } else if (m.getSeverity()==ResultSeverityEnum.WARNING) warn++;
      }
      System.out.println(f + "\t" + err + "\t" + warn + "\t" + ms + "\t" + msgs);
    }
  }
}
