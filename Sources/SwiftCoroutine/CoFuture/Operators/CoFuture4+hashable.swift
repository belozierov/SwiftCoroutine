//
//  CoFuture4+hashable.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 16.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoFuture: Hashable {
    
    // MARK: - hashable
    
    @inlinable public static func == (lhs: CoFuture, rhs: CoFuture) -> Bool {
        lhs === rhs
    }
    
    @inlinable public func hash(into hasher: inout Hasher) {
        ObjectIdentifier(self).hash(into: &hasher)
    }
    
}
