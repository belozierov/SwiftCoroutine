//
//  CoChannel+Combine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 06.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if canImport(Combine)
import Combine

@available(OSX 10.15, iOS 13.0, *)
extension CoChannel: Cancellable {}

#endif
