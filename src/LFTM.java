import models.LFDMM;
import models.LFLDA;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import utility.CmdArgs;
import eval.ClusteringEval;

/**
 * Implementations of the LF-LDA and LF-DMM latent feature topic models, using collapsed Gibbs
 * sampling, as described in:
 * 
 * Dat Quoc Nguyen, Richard Billingsley, Lan Du and Mark Johnson. 2015. Improving Topic Models with
 * Latent Feature Word Representations. Transactions of the Association for Computational
 * Linguistics, vol. 3, pp. 299-313.
 * 
 * @author Dat Quoc Nguyen
 * 
 */
public class LFTM
{
    public static void main(String[] args)
    {
        CmdArgs cmdArgs = new CmdArgs();
        CmdLineParser parser = new CmdLineParser(cmdArgs);
        try {

            parser.parseArgument(args);

            if (cmdArgs.model.equals("LFLDA")) {
                LFLDA lflda = new LFLDA(cmdArgs.corpus, cmdArgs.vectors, cmdArgs.ntopics,
                        cmdArgs.alpha, cmdArgs.beta, cmdArgs.lambda, cmdArgs.initers,
                        cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
                        cmdArgs.initTopicAssgns, cmdArgs.savestep);
                lflda.inference();
            }
            else if (cmdArgs.model.equals("LFDMM")) {
                LFDMM lfdmm = new LFDMM(cmdArgs.corpus, cmdArgs.vectors, cmdArgs.ntopics,
                        cmdArgs.alpha, cmdArgs.beta, cmdArgs.lambda, cmdArgs.initers,
                        cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
                        cmdArgs.initTopicAssgns, cmdArgs.savestep);
                lfdmm.inference();
            }
            else if (cmdArgs.model.equals("Eval")) {
                ClusteringEval.evaluate(cmdArgs.labelFile, cmdArgs.dir, cmdArgs.prob);
            }
            else {
                System.out
                        .println("Error: Option \"-model\" must get \"LFLDA\" or \"LFDMM\" or \"Eval\"");
                System.out.println("\tLFLDA: Specify the LF-LDA topic model");
                System.out.println("\tLFDMM: Specify the LF-DMM topic model");
                System.out.println("\tEval: Specify the document clustering evaluation");
                help(parser);
                return;
            }
        }
        catch (CmdLineException cle) {
            System.out.println("Error: " + cle.getMessage());
            help(parser);
            return;
        }
        catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            e.printStackTrace();
            return;
        }
    }

    public static void help(CmdLineParser parser)
    {
        System.out.println("java -jar LFTM.jar [options ...] [arguments...]");
        parser.printUsage(System.out);
    }
}
