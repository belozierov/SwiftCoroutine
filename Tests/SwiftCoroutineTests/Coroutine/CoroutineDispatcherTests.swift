//
//  CoroutineDispatcherTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 10.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoroutineDispatcherTests: XCTestCase {
    
//    func testSharedCoroutineDispatcherDeinit() {
//        let group = DispatchGroup()
//        var dispatcher: SharedCoroutineDispatcher! =
//            SharedCoroutineDispatcher(contextsCount: 1, stackSize: 32 * 1024)
//        weak var referance = dispatcher
//        group.enter()
//        dispatcher.execute(on: DispatchQueue.global()) {
//            dispatcher = nil
//            DispatchQueue.global().asyncAfter(wallDeadline: .now() + 1, execute: group.leave)
//        }
//        group.wait()
//        XCTAssertNil(referance)
//    }
    
}
