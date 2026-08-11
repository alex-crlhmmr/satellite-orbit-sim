package sim.orbit.validation;

import java.io.File;
import java.util.Locale;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel;
import org.orekit.forces.gravity.NewtonianAttraction;
import org.orekit.forces.gravity.ThirdBodyAttraction;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider;
import org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient;
import org.orekit.forces.radiation.SolarRadiationPressure;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.OrbitType;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.numerical.NumericalPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.PVCoordinates;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.Constants;

/** Minimal, independent Orekit Cartesian truth generator. */
public final class OrekitReference {
    // Brahe's IERS/ICGEM-compatible Earth GM; model parity is essential for
    // measuring implementation error rather than constant-selection error.
    private static final double MU = 3.986004415e14;

    private OrekitReference() {}

    public static void main(String[] args) {
        if (args.length != 15) {
            throw new IllegalArgumentException(
                "usage: data-dir profile gravity-file degree order epoch duration sample " +
                "x y z vx vy vz output-frame");
        }
        Locale.setDefault(Locale.ROOT);
        File dataDir = new File(args[0]);
        String profile = args[1];
        String gravityFile = args[2];
        int degree = Integer.parseInt(args[3]);
        int order = Integer.parseInt(args[4]);
        double duration = Double.parseDouble(args[6]);
        double sample = Double.parseDouble(args[7]);
        Vector3D p = new Vector3D(Double.parseDouble(args[8]), Double.parseDouble(args[9]), Double.parseDouble(args[10]));
        Vector3D v = new Vector3D(Double.parseDouble(args[11]), Double.parseDouble(args[12]), Double.parseDouble(args[13]));

        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(dataDir));
        if (!"-".equals(gravityFile)) {
            DataContext.getDefault().getDataProvidersManager().addProvider(
                new DirectoryCrawler(new File(gravityFile).getParentFile()));
        }
        AbsoluteDate epoch = new AbsoluteDate(args[5], TimeScalesFactory.getUTC());

        Frame gcrf = FramesFactory.getGCRF();
        CartesianOrbit initial = new CartesianOrbit(new PVCoordinates(p, v), gcrf, epoch, MU);
        DormandPrince853Integrator integrator = new DormandPrince853Integrator(1.0e-6, 60.0, 1.0e-8, 1.0e-14);
        NumericalPropagator propagator = new NumericalPropagator(integrator);
        propagator.setOrbitType(OrbitType.CARTESIAN);
        propagator.setInitialState(new SpacecraftState(initial, 100.0));
        propagator.removeForceModels();

        if ("gravity".equals(profile)) {
            GravityFieldFactory.clearPotentialCoefficientsReaders();
            GravityFieldFactory.addPotentialCoefficientsReader(
                new org.orekit.forces.gravity.potential.ICGEMFormatReader(
                    new File(gravityFile).getName(), false));
            NormalizedSphericalHarmonicsProvider provider =
                GravityFieldFactory.getNormalizedProvider(degree, order);
            Frame itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, true);
            propagator.addForceModel(new HolmesFeatherstoneAttractionModel(itrf, provider));
        } else if ("two_body".equals(profile)) {
            propagator.addForceModel(new NewtonianAttraction(MU));
        } else if ("third_body".equals(profile)) {
            propagator.addForceModel(new NewtonianAttraction(MU));
            propagator.addForceModel(new ThirdBodyAttraction(CelestialBodyFactory.getSun()));
            propagator.addForceModel(new ThirdBodyAttraction(CelestialBodyFactory.getMoon()));
        } else if ("srp".equals(profile) || "srp_sunlight".equals(profile)) {
            propagator.addForceModel(new NewtonianAttraction(MU));
            Frame itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, true);
            OneAxisEllipsoid earth = new OneAxisEllipsoid(
                "srp_sunlight".equals(profile) ? 1.0 : Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                "srp_sunlight".equals(profile) ? 0.0 : Constants.WGS84_EARTH_FLATTENING,
                itrf);
            propagator.addForceModel(new SolarRadiationPressure(
                CelestialBodyFactory.getSun(), earth,
                new IsotropicRadiationSingleCoefficient(1.0, 1.5)));
        } else {
            throw new IllegalArgumentException("unknown profile: " + profile);
        }

        System.out.println("elapsed_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps");
        long count = Math.round(Math.floor(duration / sample));
        for (long k = 0; k <= count; ++k) {
            double elapsed = k * sample;
            SpacecraftState state = propagator.propagate(epoch.shiftedBy(elapsed));
            PVCoordinates pv = state.getPVCoordinates(gcrf);
            Vector3D rp = pv.getPosition();
            Vector3D rv = pv.getVelocity();
            System.out.printf(Locale.ROOT, "%.9f,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e%n",
                elapsed, rp.getX(), rp.getY(), rp.getZ(), rv.getX(), rv.getY(), rv.getZ());
        }
    }
}
