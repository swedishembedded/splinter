// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose model servers on a
// cluster speak TLS to each other and to their clients with no certificate
// infrastructure to set up, for its clients. If your team needs expertise
// in securing model traffic across the nodes of an allocation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The certificates a brain fleet serves TLS with.
//!
//! A fleet makes its own certificate authority when it starts, issues one
//! server certificate per node (its name and its address as subject
//! alternative names), and discards the authority's key: no further
//! certificate can ever chain to it. The authority is also name-constrained
//! to the fleet's nodes, so even its certificate vouches for nothing else.
//! The clients of this process trust it (and only it adds to their roots);
//! everything else stays verified as before.

use std::net::IpAddr;

use rcgen::{
    BasicConstraints, CertificateParams, CidrSubnet, DistinguishedName, DnType,
    ExtendedKeyUsagePurpose, GeneralSubtree, IsCa, Issuer, KeyPair, KeyUsagePurpose,
    NameConstraints, SanType,
};

use crate::error::PolicyError;

/// A node's server identity: its certificate chain and private key, as PEM.
pub struct ServerIdentity {
    /// The node's certificate, then the authority's.
    pub chain_pem: String,
    /// The node's private key (PKCS #8).
    pub key_pem: String,
}

/// The authority's certificate and one identity per node, in `nodes`' order.
pub struct FleetCertificates {
    /// The authority every server certificate chains to, as PEM.
    pub authority_pem: String,
    /// The servers' identities.
    pub servers: Vec<ServerIdentity>,
}

fn refused(reason: String) -> PolicyError {
    PolicyError::Endpoint {
        url: "fleet".into(),
        reason: format!("making the fleet's certificates: {reason}"),
    }
}

/// Issues the fleet's certificates for `nodes` (name and address of each).
/// The authority's key lives only for the length of this call.
///
/// # Errors
/// A name is not usable as a certificate name, or signing failed.
pub fn issue(nodes: &[(String, IpAddr)]) -> Result<FleetCertificates, PolicyError> {
    let authority_key = KeyPair::generate().map_err(|e| refused(e.to_string()))?;
    let mut authority = CertificateParams::new(Vec::new()).map_err(|e| refused(e.to_string()))?;
    authority.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
    authority.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
    let mut name = DistinguishedName::new();
    name.push(DnType::CommonName, "splinter brain fleet");
    authority.distinguished_name = name;
    authority.name_constraints = Some(NameConstraints {
        permitted_subtrees: nodes
            .iter()
            .flat_map(|(node, address)| {
                [
                    GeneralSubtree::DnsName(node.clone()),
                    GeneralSubtree::IpAddress(CidrSubnet::from_addr_prefix(
                        *address,
                        if address.is_ipv4() { 32 } else { 128 },
                    )),
                ]
            })
            .collect(),
        excluded_subtrees: Vec::new(),
    });
    let authority_cert = authority
        .self_signed(&authority_key)
        .map_err(|e| refused(e.to_string()))?;
    let authority_pem = authority_cert.pem();
    let issuer = Issuer::new(authority, authority_key);
    let mut servers = Vec::with_capacity(nodes.len());
    for (node, address) in nodes {
        let key = KeyPair::generate().map_err(|e| refused(e.to_string()))?;
        let mut params = CertificateParams::new(vec![node.clone()])
            .map_err(|e| refused(format!("{node}: {e}")))?;
        params.subject_alt_names.push(SanType::IpAddress(*address));
        // What a strict verifier (OpenSSL's X.509 strict mode, Python's
        // default since 3.13) asks of a server certificate.
        params.use_authority_key_identifier_extension = true;
        params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
        let mut subject = DistinguishedName::new();
        subject.push(DnType::CommonName, node.as_str());
        params.distinguished_name = subject;
        let cert = params
            .signed_by(&key, &issuer)
            .map_err(|e| refused(format!("{node}: {e}")))?;
        servers.push(ServerIdentity {
            chain_pem: format!("{}{}", cert.pem(), authority_pem),
            key_pem: key.serialize_pem(),
        });
    }
    Ok(FleetCertificates {
        authority_pem,
        servers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_node_gets_a_chain_to_one_authority_whose_key_is_not_kept() {
        let issued = issue(&[
            ("n442".into(), "10.150.4.178".parse().unwrap()),
            ("n467".into(), "10.150.4.240".parse().unwrap()),
        ])
        .unwrap();
        assert_eq!(issued.servers.len(), 2);
        assert!(issued.authority_pem.contains("BEGIN CERTIFICATE"));
        for server in &issued.servers {
            assert_eq!(server.chain_pem.matches("BEGIN CERTIFICATE").count(), 2);
            assert!(server.chain_pem.ends_with(&issued.authority_pem));
            assert!(server.key_pem.contains("PRIVATE KEY"));
        }
        assert_ne!(issued.servers[0].key_pem, issued.servers[1].key_pem);
    }
}
